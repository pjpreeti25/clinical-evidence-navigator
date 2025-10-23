#!/usr/bin/env python3
import os
import sqlite3
import logging
from datetime import datetime
from typing import Any, Dict, Optional
import json
import time

# Core dependencies
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from groq import Groq

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ClinicalEvidenceNavigator")

# ----------------------------
# Small helpers
# ----------------------------
def _coerce_query(arg: str) -> str:
    """Accept either a plain string or a JSON string like {"query": "..."} from crew tools."""
    try:
        if isinstance(arg, str) and arg.strip().startswith("{"):
            return (json.loads(arg) or {}).get("query", arg)
    except Exception:
        pass
    return arg


# ============================================================
# Main Navigator - Cloud Version
# ============================================================
class ClinicalEvidenceNavigator:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.conn: Optional[sqlite3.Connection] = None
        self.collection = None
        self.model_name: Optional[str] = None
        self.crewai_llm: Optional[LLM] = None
        self.groq_client: Optional[Groq] = None

        self.setup_database()
        self.setup_vector_store()
        self.setup_groq()
        self.setup_agents()

    # ----------------------------
    # SQLite
    # ----------------------------
    def setup_database(self):
        """Initialize SQLite database for metadata"""
        self.db_path = os.path.join(self.data_dir, "clinical_evidence.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                pmid TEXT,
                nct_id TEXT,
                source TEXT,
                publication_date TEXT,
                study_type TEXT,
                url TEXT,
                abstract TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                gap_description TEXT,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()
        logger.info("Database initialized successfully")

    # ----------------------------
    # Chroma
    # ----------------------------
    def setup_vector_store(self):
        """Initialize Chroma vector database"""
        chroma_path = os.path.join(self.data_dir, "chroma_db")
        os.makedirs(chroma_path, exist_ok=True)

        # Disable telemetry to avoid noisy logs
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="clinical_evidence",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store initialized successfully")

    # ----------------------------
    def setup_groq(self):
    """Initialize Groq API client and create CrewAI LLM binding."""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            print(f"API Key found: {api_key[:10]}..." if api_key else "No API key")
            if not api_key:
                logger.warning("GROQ_API_KEY not found")
                return
            
            self.groq_client = Groq(api_key=api_key)
            print("Groq client created successfully")
        
        # Test the API
            test_response = self.groq_client.chat.completions.create(
               model="llama-3.1-8b-instant",
               messages=[{"role": "user", "content": "Hello"}],
               max_tokens=10 )
            print("Test API call successful")
        
        # Set model name
            self.model_name = "llama-3.1-8b-instant"
        
        # Create CrewAI LLM binding for Groq
           self.crewai_llm = LLM(
              model="groq/llama-3.1-8b-instant",
              api_key=api_key,
              temperature=0.2,
              timeout=60,
              max_tokens=1024,
           )

           logger.info("Groq API initialized with model: %s", self.model_name)
        
        except Exception as e:
           print(f"Groq setup error: {e}")
           logger.error(f"Groq setup failed: {e}")
           self.model_name = None
           self.crewai_llm = None
           self.groq_client = None

    # ----------------------------
    # Persistence helpers
    # ----------------------------
    def upsert_pubmed_record(
        self,
        pmid: str,
        title: str,
        abstract: str,
        pub_date: Optional[str],
        url: str,
    ):
        if not self.conn or not self.collection:
            return

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO studies(title, pmid, source, publication_date, study_type, url, abstract)
            VALUES (?, ?, 'PubMed', ?, 'Article', ?, ?)
            """,
            (title, pmid, pub_date or None, url, abstract),
        )
        self.conn.commit()

        doc_id = f"pmid:{pmid}"
        meta = {
            "source": "PubMed",
            "pmid": pmid,
            "title": title,
            "url": url,
            "publication_date": pub_date,
        }
        try:
            self.collection.add(
                ids=[doc_id],
                documents=[f"{title}\n\n{abstract}"],
                metadatas=[meta],
            )
        except Exception as e:
            logger.warning("Chroma add failed for pmid:%s - %s", pmid, e)

    def upsert_trial_record(
        self,
        nct_id: str,
        title: str,
        summary: str,
        status: str,
        phase: str,
        url: str,
    ):
        if not self.conn or not self.collection:
            return

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO studies(title, nct_id, source, study_type, url, abstract)
            VALUES (?, ?, 'ClinicalTrials', 'Clinical Trial', ?, ?)
            """,
            (title, nct_id, url, summary),
        )
        self.conn.commit()

        doc_id = f"nct:{nct_id}"
        meta = {
            "source": "ClinicalTrials",
            "nct_id": nct_id,
            "title": title,
            "url": url,
            "phase": phase,
            "status": status,
        }
        try:
            self.collection.add(
                ids=[doc_id],
                documents=[f"{title}\n\n{summary}"],
                metadatas=[meta],
            )
        except Exception as e:
            logger.warning("Chroma add failed for nct:%s - %s", nct_id, e)

    # ----------------------------
    # CrewAI Agents
    # ----------------------------
    def setup_agents(self):
        """Initialize CrewAI agents and tools"""

        # Tools
        pubmed_tool = PubMedScraperTool()
        trials_tool = ClinicalTrialsScraperTool()
        analysis_tool = EvidenceAnalysisTool()
        synthesis_tool = EvidenceSynthesisTool()
        gap_tool = GapAnalysisTool()

        # tool backrefs
        for t in (pubmed_tool, trials_tool):
            t.navigator = self  # set at runtime

        # pass groq client to LLM-using tools
        analysis_tool.groq_client = self.groq_client
        synthesis_tool.groq_client = self.groq_client
        gap_tool.groq_client = self.groq_client

        # Agents (bind CrewAI LLM so the agent can plan/execute with Groq)
        self.research_agent = Agent(
            role="Clinical Research Specialist",
            goal="Find and extract relevant clinical evidence from PubMed and ClinicalTrials.gov",
            backstory=(
                "You are an expert clinical researcher with deep knowledge of medical databases "
                "and evidence-based medicine. You excel at finding relevant studies and extracting key information."
            ),
            tools=[pubmed_tool, trials_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.crewai_llm,
        )

        self.analysis_agent = Agent(
            role="Evidence Analysis Expert",
            goal="Analyze clinical evidence quality, identify patterns, and assess study reliability",
            backstory=(
                "You are a biostatistician and evidence synthesis expert who specializes in critical appraisal of "
                "clinical studies and meta-analysis."
            ),
            tools=[analysis_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.crewai_llm,
        )

        self.synthesis_agent = Agent(
            role="Clinical Evidence Synthesizer",
            goal="Synthesize evidence into actionable clinical recommendations and identify knowledge gaps",
            backstory=(
                "You are a clinical guidelines expert who translates research evidence into practical "
                "recommendations for healthcare providers."
            ),
            tools=[synthesis_tool, gap_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.crewai_llm,
        )

    # ----------------------------
    # Orchestrate query (agent) with constraints + fallback
    # ----------------------------
    def query_evidence(self, clinical_query: str) -> Dict[str, Any]:
        """Main method to process clinical evidence queries"""
        logger.info("Processing query: %s", clinical_query)

        research_task = Task(
            description=f"""
Search for clinical evidence related to: {clinical_query}

You MUST use only these tools: pubmed_scraper, clinicaltrials_scraper.
- First call pubmed_scraper with the full query.
- Then call clinicaltrials_scraper with the full query.
If a tool errors, DO NOT invent new tools or "Manual Search". Proceed with what you have.

Extract and return titles, abstracts, PMIDs/NCT IDs, dates, study types, key findings.
            """.strip(),
            agent=self.research_agent,
            expected_output="JSON list of studies/trials with metadata",
            max_iter=1,
            human_input=False,
        )

        analysis_task = Task(
            description=f"""
Analyze the clinical evidence found for: {clinical_query}

Evaluate:
- Study quality and methodology
- Risk of bias and limitations
- Statistical significance and clinical significance
- Consistency across studies
- Strength of evidence levels

Provide evidence grading using standard frameworks.
            """.strip(),
            agent=self.analysis_agent,
            expected_output="Evidence quality assessment with strength ratings",
            max_iter=1,
            human_input=False,
        )

        synthesis_task = Task(
            description=f"""
Synthesize evidence and create clinical recommendations for: {clinical_query}

Provide:
- EXECUTIVE SUMMARY
- KEY FINDINGS
- CLINICAL RECOMMENDATIONS (with evidence grades A/B/C/D)
- PRACTICE IMPLICATIONS
- LIMITATIONS AND CONSIDERATIONS
- EVIDENCE GAPS and future research suggestions

Format as a structured clinical evidence report.
            """.strip(),
            agent=self.synthesis_agent,
            expected_output="Comprehensive clinical evidence report with recommendations",
            max_iter=1,
            human_input=False,
        )

        try:
            crew = Crew(
                agents=[self.research_agent, self.analysis_agent, self.synthesis_agent],
                tasks=[research_task, analysis_task, synthesis_task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            self.store_results(clinical_query, str(result))

            return {
                "query": clinical_query,
                "timestamp": datetime.now().isoformat(),
                "results": str(result),
                "evidence_count": self.get_evidence_count(clinical_query),
            }
        except Exception as e:
            logger.exception("Crew failed, falling back to deterministic pipeline: %s", e)
            return self.query_evidence_fast(clinical_query)

    # ----------------------------
    # Deterministic pipeline (no planning, no loops)
    # ----------------------------
    def query_evidence_fast(self, clinical_query: str) -> Dict[str, Any]:
        """Deterministic pipeline: scrape -> analyze -> synthesize -> gaps (no CrewAI planning)."""
        # Tools with backrefs so they persist to DB/Chroma
        pub = PubMedScraperTool(); pub.navigator = self
        ct = ClinicalTrialsScraperTool(); ct.navigator = self

        pub_res_raw = pub._run(clinical_query)
        ct_res_raw = ct._run(clinical_query)

        evidence_text = f"PUBMED:\n{pub_res_raw}\n\nCLINICALTRIALS:\n{ct_res_raw}"

        ana = EvidenceAnalysisTool(); ana.groq_client = self.groq_client
        syn = EvidenceSynthesisTool(); syn.groq_client = self.groq_client
        gap = GapAnalysisTool(); gap.groq_client = self.groq_client

        analyzed = ana._run(evidence_text)
        synthesized = syn._run(analyzed)
        gaps = gap._run(synthesized)

        full_report = f"{synthesized}\n\n---\nGAP ANALYSIS\n{gaps}"

        self.store_results(clinical_query, full_report)

        return {
            "query": clinical_query,
            "timestamp": datetime.now().isoformat(),
            "results": full_report,
            "evidence_count": self.get_evidence_count(clinical_query),
        }

    def store_results(self, query: str, results: str):
        """Store analysis results in database"""
        if not self.conn:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO evidence_gaps (query, gap_description, recommendations)
            VALUES (?, ?, ?)
            """,
            (query, "Analysis completed", results),
        )
        self.conn.commit()

    def get_evidence_count(self, query: str) -> int:
        """Get count of evidence pieces for a query (based on nearest docs present)"""
        try:
            search_results = self.collection.query(query_texts=[query], n_results=50)
            docs = search_results.get("documents") or []
            return len(docs[0]) if docs and docs[0] else 0
        except Exception:
            return 0


# ============================================================
# Custom Tools for CrewAI Agents
# ============================================================

class PubMedScraperTool(BaseTool):
    name: str = "pubmed_scraper"
    description: str = "Scrapes PubMed for clinical research papers"
    navigator: Optional[Any] = None  # backref set at runtime
    model_config = {"extra": "allow"}  # allow extra attributes

    def _run(self, query: str) -> str:
        """Scrape PubMed for research papers and persist to DB + Chroma"""
        try:
            query = _coerce_query(query)
            headers = {"User-Agent": "ClinicalEvidenceNavigator/1.0"}

            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": 20,
                "retmode": "json",
                "sort": "relevance",
            }
            data = requests.get(base_url, params=params, timeout=30, headers=headers).json()
            if "esearchresult" not in data or "idlist" not in data["esearchresult"]:
                return json.dumps([])

            pmids = data["esearchresult"]["idlist"][:10]
            if not pmids:
                return json.dumps([])

            details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            details_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
            details_response = requests.get(details_url, params=details_params, timeout=60, headers=headers)

            soup = BeautifulSoup(details_response.content, "xml")
            results = []

            for article in soup.find_all("PubmedArticle")[:5]:
                try:
                    title_elem = article.find("ArticleTitle")
                    abstract_elem = article.find("AbstractText")
                    pmid_elem = article.find("PMID")

                    title = title_elem.text if title_elem else "No title"
                    abstract = abstract_elem.text if abstract_elem else "No abstract"
                    pmid = pmid_elem.text if pmid_elem else "No PMID"

                    pub_date = None
                    dp = article.find("DateCompleted") or article.find("PubDate")
                    if dp:
                        yy = dp.find("Year").text if dp.find("Year") else None
                        mm = dp.find("Month").text if dp.find("Month") else None
                        dd = dp.find("Day").text if dp.find("Day") else None
                        pub_date = "-".join([p for p in [yy, mm, dd] if p])

                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                    item = {
                        "pmid": pmid,
                        "title": title,
                        "abstract": (abstract[:500] + "...") if len(abstract) > 500 else abstract,
                        "publication_date": pub_date,
                        "url": url,
                        "source": "PubMed",
                    }
                    results.append(item)

                    if self.navigator and pmid != "No PMID":
                        self.navigator.upsert_pubmed_record(
                            pmid=pmid,
                            title=title,
                            abstract=abstract,
                            pub_date=pub_date,
                            url=url,
                        )
                except Exception:
                    continue

            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error scraping PubMed: {str(e)}"


class ClinicalTrialsScraperTool(BaseTool):
    name: str = "clinicaltrials_scraper"
    description: str = "Scrapes ClinicalTrials.gov for ongoing and completed clinical trials"

    navigator: Optional[Any] = None
    model_config = {"extra": "allow"}

    def _run(self, query: str) -> str:
        """Scrape ClinicalTrials.gov for clinical trials and persist to DB + Chroma"""
        try:
            query = _coerce_query(query)
            headers = {"User-Agent": "ClinicalEvidenceNavigator/1.0"}

            base_url = "https://clinicaltrials.gov/api/query/study_fields"
            params = {
                "expr": query,
                "fields": "NCTId,BriefTitle,Condition,Phase,OverallStatus,StudyType,BriefSummary",
                "max_rnk": 10,
                "fmt": "json",
            }
            resp = requests.get(base_url, params=params, timeout=30, headers=headers)

            if not resp.ok:
                time.sleep(1.0)
                resp = requests.get(base_url, params=params, timeout=30, headers=headers)

            try:
                data = resp.json()
            except ValueError:
                return json.dumps({
                    "error": "ClinicalTrials.gov did not return JSON",
                    "status": resp.status_code,
                    "preview": resp.text[:200]
                })

            SFR = data.get("StudyFieldsResponse", {})
            studies = SFR.get("StudyFields", []) or []
            if not studies:
                return json.dumps([])

            results = []
            for study in studies[:5]:
                try:
                    nct_id = (study.get("NCTId") or [""])[0]
                    title = (study.get("BriefTitle") or [""])[0]
                    cond = (study.get("Condition") or [""])[0]
                    phase = (study.get("Phase") or [""])[0]
                    status = (study.get("OverallStatus") or [""])[0]
                    stype = (study.get("StudyType") or [""])[0]
                    summary_full = (study.get("BriefSummary") or [""])[0]
                    summary = (summary_full[:300] + "...") if summary_full else ""
                    url = f"https://clinicaltrials.gov/study/{nct_id}"

                    item = {
                        "nct_id": nct_id,
                        "title": title,
                        "condition": cond,
                        "phase": phase,
                        "status": status,
                        "study_type": stype,
                        "summary": summary,
                        "source": "ClinicalTrials.gov",
                        "url": url,
                    }
                    results.append(item)

                    if self.navigator and nct_id:
                        self.navigator.upsert_trial_record(
                            nct_id=nct_id,
                            title=title,
                            summary=summary_full or "",
                            status=status,
                            phase=phase,
                            url=url,
                        )
                except Exception:
                    continue

            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error scraping ClinicalTrials.gov: {str(e)}"


# ============================================================
# LLM-backed Analysis / Synthesis Tools - Groq Version
# ============================================================

def _extract_groq_text(response: Any) -> str:
    """Robustly extract text from Groq response."""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if content:
                return content
        content = response.get("content")
        if content:
            return content
    return "No content returned by LLM"


class EvidenceAnalysisTool(BaseTool):
    name: str = "evidence_analyzer"
    description: str = "Analyzes clinical evidence quality and provides evidence grading"

    # Declare pydantic fields; allow extras
    groq_client: Optional[Groq] = None
    model_config = {"extra": "allow"}

    def _run(self, evidence_data: str) -> str:
        """Analyze evidence quality using Groq API"""
        try:
            if not self.groq_client:
                return "Groq API not available for analysis"

            prompt = f"""
Analyze the following clinical evidence and provide:
1. Evidence quality assessment (High/Moderate/Low/Very Low)
2. Study design evaluation
3. Risk of bias assessment
4. Clinical significance
5. Recommendations strength grading

Evidence data:
{evidence_data}

Provide a structured analysis using evidence-based medicine principles.
            """.strip()

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return _extract_groq_text(response)
        except Exception as e:
            return f"Error analyzing evidence: {str(e)}"


class EvidenceSynthesisTool(BaseTool):
    name: str = "evidence_synthesizer"
    description: str = "Synthesizes evidence into clinical recommendations"

    groq_client: Optional[Groq] = None
    model_config = {"extra": "allow"}

    def _run(self, analyzed_evidence: str) -> str:
        """Synthesize evidence using Groq API"""
        try:
            if not self.groq_client:
                return "Groq API not available for synthesis"

            prompt = f"""
Based on the analyzed clinical evidence, create a comprehensive clinical report with:

1. EXECUTIVE SUMMARY
2. KEY FINDINGS
3. CLINICAL RECOMMENDATIONS (with evidence grades A/B/C/D)
4. PRACTICE IMPLICATIONS
5. LIMITATIONS AND CONSIDERATIONS

Analyzed evidence:
{analyzed_evidence}

Format as a professional clinical evidence report.
            """.strip()

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return _extract_groq_text(response)
        except Exception as e:
            return f"Error synthesizing evidence: {str(e)}"


class GapAnalysisTool(BaseTool):
    name: str = "gap_analyzer"
    description: str = "Identifies evidence gaps and research needs"

    groq_client: Optional[Groq] = None
    model_config = {"extra": "allow"}

    def _run(self, synthesized_evidence: str) -> str:
        """Identify evidence gaps using Groq API"""
        try:
            if not self.groq_client:
                return "Groq API not available for gap analysis"

            prompt = f"""
Based on the synthesized clinical evidence, identify:

1. EVIDENCE GAPS - What's missing from current research
2. RESEARCH PRIORITIES - What studies are needed
3. CLINICAL UNCERTAINTIES - Unanswered questions
4. FUTURE RESEARCH RECOMMENDATIONS

Synthesized evidence:
{synthesized_evidence}

Provide specific, actionable gap analysis.
            """.strip()

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return _extract_groq_text(response)
        except Exception as e:
            return f"Error analyzing gaps: {str(e)}"


# ============================================================
# CLI
# ============================================================
def main():
    """Main application entry point"""
    print(" Clinical Evidence Navigator - Cloud Version Starting...")

    navigator = ClinicalEvidenceNavigator()

    print("\n System initialized successfully!")
    if not navigator.model_name:
        print("Groq API not configured. Please set GROQ_API_KEY environment variable.")
    print("Available commands:")
    print("  - search: Query clinical evidence (agent)")
    print("  - search-fast: Query via deterministic pipeline (no agent)")
    print("  - quit: Exit the application")

    while True:
        try:
            command = input("\n🔍 Enter command (search/search-fast/quit): ").strip().lower()

            if command == "quit":
                print("Goodbye!")
                break

            elif command in ("search", "search-fast"):
                query = input("Enter your clinical query: ").strip()
                if query:
                    print(f"\n Searching for evidence on: {query}")
                    print("This may take a minute...")

                    if command == "search-fast":
                        results = navigator.query_evidence_fast(query)
                    else:
                        results = navigator.query_evidence(query)

                    print("\nResults Summary:")
                    print(f"   Query: {results['query']}")
                    print(f"   Timestamp: {results['timestamp']}")
                    print(f"   Evidence pieces (indexed): {results['evidence_count']}")
                    print(f"\n Detailed Results:\n{results['results']}")
                else:
                    print("Please enter a valid query")

            else:
                print("Unknown command. Use 'search', 'search-fast' or 'quit'")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Application error")



if __name__ == "__main__":
    main()




