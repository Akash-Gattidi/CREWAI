import os
import sys
import io
import json
import pandas as pd
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

# Set up Gemini API Key from environment variable
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from crewai import LLM, Agent, Task, Crew, Process
from crewai.flow.flow import Flow, listen, start, router, and_, or_
from crewai.tools import tool
from duckduckgo_search import DDGS

# Initialize FastAPI application
app = FastAPI(
    title="CREWAI - 10 Autonomous AI Agents & Flows API",
    description="Full implementation of the 10 CrewAI projects (Crews and Flows) from crewai_projects.html powered by Gemini 3.1 Flash Lite Preview.",
    version="1.0.0"
)

# Enable CORS for webapp integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# LLM Helper
# -----------------------------------------------------------------------------
def get_llm(custom_key: Optional[str] = None) -> LLM:
    """Return the specified Gemini 3.1 Flash Lite Preview LLM instance."""
    api_key = custom_key or os.getenv("GEMINI_API_KEY") or GEMINI_KEY
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not configured. Please set GEMINI_API_KEY in your environment or provide 'x-gemini-api-key' header."
        )
    return LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3
    )

# -----------------------------------------------------------------------------
# Common Search Tool
# -----------------------------------------------------------------------------
@tool("duckduckgo_search")
def duckduckgo_search_tool(query: str) -> str:
    """Search the web for up-to-date information on a specific topic."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No web results found."
        snippets = [f"Title: {r.get('title')}\nSnippet: {r.get('body')}" for r in results]
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Search temporarily unavailable: {str(e)}"

# =============================================================================
# Root & Health Endpoints
# =============================================================================
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "CrewAI 10 Projects API",
        "model": "gemini-3.1-flash-lite-preview",
        "routes": {
            "/agent1": "01 Blog Post Crew (Sequential)",
            "/agent2": "02 Competitor Analysis Crew (Hierarchical)",
            "/agent3": "03 Meeting Notes Crew (Parallel Tasks)",
            "/agent4": "04 Data Report Crew (Structured Pydantic Output)",
            "/agent5": "05 Customer Support Flow (Router)",
            "/agent6": "06 Resume Screening Flow (Router)",
            "/agent7": "07 Bug Triage Flow (Router + Crews)",
            "/agent8": "08 Content Repurposing Flow (and_ Multi-Crew)",
            "/agent9": "09 Lead Qualification Flow (Router + Crews)",
            "/agent10": "10 Approval Report Flow (Human Feedback Loop)"
        },
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "crewai-api"}

# =============================================================================
# 01 - Blog Post Crew (Sequential)
# =============================================================================
class BlogPostRequest(BaseModel):
    topic: str = Field(..., example="AI in precision farming")

@app.post("/agent1", summary="01 - Blog Post Crew (Sequential)")
def run_agent1(request: BlogPostRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 01: Blog Post Crew
    Sequential workflow: Researcher (DuckDuckGo search) -> Writer (600-word draft) -> Editor (polishes tone/facts).
    """
    try:
        llm = get_llm(x_gemini_api_key)

        researcher = Agent(
            role="Senior Research Analyst",
            goal=f"Conduct thorough research and find key data points and trends on: {request.topic}",
            backstory="You are an expert researcher with a keen eye for cutting-edge facts, statistics, and verifiable sources.",
            llm=llm,
            tools=[duckduckgo_search_tool],
            verbose=False
        )

        writer = Agent(
            role="Professional Content Writer",
            goal=f"Draft an engaging, high-value 600-word blog post based on the research findings on {request.topic}",
            backstory="You are a seasoned technology and industry writer skilled at crafting captivating narratives with clear headings.",
            llm=llm,
            verbose=False
        )

        editor = Agent(
            role="Managing Editor",
            goal="Refine, fact-check, and polish the blog draft for flawless tone, readability, and structural flow.",
            backstory="You are a meticulous editor known for perfecting tone, enhancing clarity, and ensuring professional editorial standards.",
            llm=llm,
            verbose=False
        )

        research_task = Task(
            description=f"Gather 3-5 key facts, statistics, and recent developments regarding: {request.topic}.",
            expected_output="A bulleted summary of verified research points with clear insights.",
            agent=researcher
        )

        writing_task = Task(
            description=f"Using the research, write an engaging 600-word blog post titled around '{request.topic}'. Include an introduction, 3 body sections with subheaders, and a conclusion.",
            expected_output="A full markdown blog post draft.",
            agent=writer
        )

        editing_task = Task(
            description="Review and edit the written blog post. Polish grammar, adjust tone to be authoritative yet accessible, and format with clean Markdown.",
            expected_output="The finalized, publication-ready markdown blog post.",
            agent=editor
        )

        crew = Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, writing_task, editing_task],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        return {
            "project": "01 - Blog Post Crew",
            "process": "sequential",
            "topic": request.topic,
            "blog_post": result.raw
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 02 - Competitor Analysis Crew (Hierarchical)
# =============================================================================
class CompetitorRequest(BaseModel):
    competitors: List[str] = Field(..., example=["Stripe", "Adyen", "PayPal"])

@app.post("/agent2", summary="02 - Competitor Analysis Crew (Hierarchical)")
def run_agent2(request: CompetitorRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 02: Competitor Analysis Crew
    Hierarchical workflow: Manager delegating to Site Scraper/Analyst, Pricing Analyst, News Analyst, and Strategist for a SWOT table.
    """
    try:
        llm = get_llm(x_gemini_api_key)
        comp_str = ", ".join(request.competitors)

        product_analyst = Agent(
            role="Product Offering Analyst",
            goal=f"Analyze features and core offerings of competitors: {comp_str}",
            backstory="You specialize in deconstructing technology products and cataloging feature matrices.",
            llm=llm,
            tools=[duckduckgo_search_tool],
            verbose=False
        )

        pricing_analyst = Agent(
            role="Pricing & Business Model Analyst",
            goal=f"Investigate pricing models, tiers, and fee structures for: {comp_str}",
            backstory="You are an expert in SaaS and fintech monetization models with deep knowledge of competitive pricing.",
            llm=llm,
            tools=[duckduckgo_search_tool],
            verbose=False
        )

        news_analyst = Agent(
            role="Market News Analyst",
            goal=f"Track recent news, announcements, and market sentiment regarding: {comp_str}",
            backstory="You monitor industry press releases, media coverage, and recent strategic milestones.",
            llm=llm,
            tools=[duckduckgo_search_tool],
            verbose=False
        )

        strategist = Agent(
            role="Chief Strategy Officer",
            goal=f"Synthesize all competitor insights into a comprehensive comparative SWOT analysis table and strategic takeaways.",
            backstory="You are an executive strategist adept at identifying market gaps, moats, and vulnerabilities.",
            llm=llm,
            verbose=False
        )

        t1 = Task(
            description=f"Analyze the product portfolios and core strengths of: {comp_str}.",
            expected_output="Product feature breakdown for each competitor.",
            agent=product_analyst
        )

        t2 = Task(
            description=f"Examine pricing models and tier structures for: {comp_str}.",
            expected_output="Pricing and commercial breakdown for each competitor.",
            agent=pricing_analyst
        )

        t3 = Task(
            description=f"Summarize the latest major news and product releases for: {comp_str}.",
            expected_output="Top recent news highlights per competitor.",
            agent=news_analyst
        )

        t4 = Task(
            description=f"Construct a comprehensive Markdown SWOT table comparing {comp_str}, followed by 3 actionable strategic recommendations.",
            expected_output="A complete Markdown competitor intelligence report featuring a detailed SWOT table.",
            agent=strategist
        )

        crew = Crew(
            agents=[product_analyst, pricing_analyst, news_analyst, strategist],
            tasks=[t1, t2, t3, t4],
            process=Process.hierarchical,
            manager_llm=llm,
            verbose=False
        )

        result = crew.kickoff()
        return {
            "project": "02 - Competitor Analysis Crew",
            "process": "hierarchical",
            "competitors": request.competitors,
            "report": result.raw
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 03 - Meeting Notes Crew (Parallel Tasks)
# =============================================================================
class MeetingNotesRequest(BaseModel):
    transcript: str = Field(..., example="Alex: We need to ship the auth feature by Friday. Priya: I will handle the token refresh API and tests by Thursday. John: I will update the frontend login view by Friday morning. Alex: Agreed. Also, we decided to drop support for IE11.")

@app.post("/agent3", summary="03 - Meeting Notes Crew (Parallel Tasks)")
def run_agent3(request: MeetingNotesRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 03: Meeting Notes Crew
    Parallel async tasks: Summarizer, Action Finder, Decision Logger run in parallel, Formatter joins and outputs notes.md.
    """
    try:
        llm = get_llm(x_gemini_api_key)

        summarizer = Agent(
            role="Executive Meeting Summarizer",
            goal="Distill meeting transcripts into an exact 5-line executive summary.",
            backstory="You excel at rapid, high-density synthesis of leadership discussions.",
            llm=llm,
            verbose=False
        )

        action_finder = Agent(
            role="Action Items Tracker",
            goal="Identify every action item with assigned Owner and Target Due Date.",
            backstory="You ensure total accountability by extracting concrete commitments from conversational transcripts.",
            llm=llm,
            verbose=False
        )

        decision_logger = Agent(
            role="Governance & Decisions Recorder",
            goal="Catalog all official decisions and policy choices agreed upon in the meeting.",
            backstory="You record decisive outcomes and consensus milestones accurately.",
            llm=llm,
            verbose=False
        )

        formatter = Agent(
            role="Documentation Specialist",
            goal="Compile the summary, action items, and decisions into a beautifully formatted meeting notes document.",
            backstory="You format organizational knowledge into crisp, standardized Markdown documentation.",
            llm=llm,
            verbose=False
        )

        t_summary = Task(
            description=f"Write a concise 5-line executive summary of this transcript:\n\n{request.transcript}",
            expected_output="An exact 5-line executive summary.",
            agent=summarizer,
            async_execution=True
        )

        t_actions = Task(
            description=f"Extract all action items from this transcript. Include Owner, Task, and Due Date:\n\n{request.transcript}",
            expected_output="A bulleted list of action items formatted as: - [ ] Task | Owner: Name | Due: Date",
            agent=action_finder,
            async_execution=True
        )

        t_decisions = Task(
            description=f"List all explicit decisions agreed upon in this transcript:\n\n{request.transcript}",
            expected_output="A bulleted list of key decisions made.",
            agent=decision_logger,
            async_execution=True
        )

        t_format = Task(
            description="Wait for all three preceding tasks (Summary, Action Items, Decisions) and format them into a polished meeting notes document titled 'Meeting Notes & Action Plan'.",
            expected_output="A complete Markdown document containing Executive Summary, Action Items Table/List, and Decisions Log.",
            agent=formatter,
            context=[t_summary, t_actions, t_decisions]
        )

        crew = Crew(
            agents=[summarizer, action_finder, decision_logger, formatter],
            tasks=[t_summary, t_actions, t_decisions, t_format],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        return {
            "project": "03 - Meeting Notes Crew",
            "process": "parallel_async_tasks",
            "notes": result.raw
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 04 - Data Report Crew (Structured Output)
# =============================================================================
class DataReportRequest(BaseModel):
    csv_data: str = Field(
        ...,
        example="product,region,units_sold,revenue\nWidget A,North,120,2400\nWidget B,South,80,3200\nWidget C,East,150,1500\nWidget D,West,200,6000\nWidget A,South,90,1800"
    )

class SalesStats(BaseModel):
    total_revenue: float = Field(..., description="Calculated sum of revenue across all rows")
    total_units_sold: int = Field(..., description="Total quantity of units sold")
    top_performing_product: str = Field(..., description="Product generating highest revenue")
    top_region: str = Field(..., description="Region with highest revenue or volume")

@app.post("/agent4", summary="04 - Data Report Crew (Structured Output)")
def run_agent4(request: DataReportRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 04: Data Report Crew
    Structured output workflow: Data Analyst (Pydantic stats) -> Insight Writer (Top 3 findings) -> Reviewer (verified report).
    """
    try:
        llm = get_llm(x_gemini_api_key)

        # Parse CSV with pandas to generate pre-computed metrics
        try:
            df = pd.read_csv(io.StringIO(request.csv_data))
            summary_stats = df.describe(include="all").to_string()
        except Exception:
            summary_stats = "Raw table provided in prompt."

        analyst = Agent(
            role="Senior Quantitative Data Analyst",
            goal="Analyze the sales data and extract structured key performance metrics.",
            backstory="You are a data expert with mastery in quantitative statistics and metric modeling.",
            llm=llm,
            verbose=False
        )

        insight_writer = Agent(
            role="Business Intelligence Storyteller",
            goal="Derive the top 3 actionable business findings and strategic growth drivers from the data.",
            backstory="You translate raw numerical metrics into compelling commercial intelligence for business leaders.",
            llm=llm,
            verbose=False
        )

        reviewer = Agent(
            role="Audit & Verification Reviewer",
            goal="Cross-check every number against the original data and formulate the final executive sales report.",
            backstory="You ensure zero hallucinations, total mathematical consistency, and authoritative reporting.",
            llm=llm,
            verbose=False
        )

        t_stats = Task(
            description=f"Analyze this sales CSV dataset:\n\n{request.csv_data}\n\nPandas Summary:\n{summary_stats}\n\nCompute and output the structured sales statistics.",
            expected_output="Structured sales statistics matching the SalesStats schema.",
            agent=analyst,
            output_pydantic=SalesStats
        )

        t_insights = Task(
            description="Based on the analyzed data and metrics, formulate the top 3 key business findings, market patterns, and growth opportunities.",
            expected_output="Top 3 business insights with clear commercial rationale.",
            agent=insight_writer
        )

        t_review = Task(
            description="Audit all figures against the dataset and compile the complete final sales analysis report in Markdown format.",
            expected_output="A verified executive sales report containing summary metrics, top 3 insights, and recommendations.",
            agent=reviewer
        )

        crew = Crew(
            agents=[analyst, insight_writer, reviewer],
            tasks=[t_stats, t_insights, t_review],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        pydantic_output = t_stats.output.pydantic.model_dump() if t_stats.output and t_stats.output.pydantic else None

        return {
            "project": "04 - Data Report Crew",
            "process": "structured_pydantic_output",
            "structured_metrics": pydantic_output,
            "final_report": result.raw
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 05 - Customer Support Flow (Router)
# =============================================================================
class SupportTicketRequest(BaseModel):
    ticket_id: str = Field(..., example="TCK-9481")
    customer_name: str = Field(..., example="Alice Smith")
    subject: str = Field(..., example="Double charged for my annual plan")
    message: str = Field(..., example="I was billed twice on my credit card for the Pro annual subscription on October 20th.")

@app.post("/agent5", summary="05 - Customer Support Flow (Router)")
def run_agent5(request: SupportTicketRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 05: Customer Support Flow
    Flow workflow: receive_ticket -> categorize_ticket -> route_by_category -> handle_billing / handle_access / handle_other -> resolve_ticket.
    """
    try:
        llm = get_llm(x_gemini_api_key)

        class CustomerSupportFlow(Flow):
            ticket_data: Dict[str, Any] = {}
            category: str = "other"
            priority: str = "normal"
            resolution: str = ""

            @start()
            def receive_ticket(self):
                self.ticket_data = {
                    "id": request.ticket_id,
                    "customer": request.customer_name,
                    "subject": request.subject,
                    "message": request.message
                }
                return self.ticket_data

            @listen(receive_ticket)
            def categorize_ticket(self, data):
                prompt = (
                    f"Classify this customer support ticket into EXACTLY ONE category: 'billing', 'account_access', or 'other'.\n"
                    f"Subject: {data['subject']}\n"
                    f"Message: {data['message']}\n\n"
                    f"Respond with only the single word: billing, account_access, or other."
                )
                resp = llm.call(messages=[{"role": "user", "content": prompt}]).strip().lower()
                if "billing" in resp:
                    self.category = "billing"
                elif "account" in resp or "access" in resp:
                    self.category = "account_access"
                else:
                    self.category = "other"
                return self.category

            @router(categorize_ticket)
            def route_by_category(self):
                return self.category

            @listen("billing")
            def handle_billing(self):
                self.priority = "high"
                prompt = (
                    f"You are a Tier 2 Billing Specialist. Formulate a polite, reassuring resolution for:\n"
                    f"Customer: {self.ticket_data['customer']}\nIssue: {self.ticket_data['subject']} - {self.ticket_data['message']}\n"
                    f"State the refund timeline, receipt verification, and apologies for the inconvenience."
                )
                self.resolution = llm.call(messages=[{"role": "user", "content": prompt}]).strip()
                return self.resolution

            @listen("account_access")
            def handle_access(self):
                self.priority = "high"
                prompt = (
                    f"You are an Account Security & Access Specialist. Formulate an empathetic, clear step-by-step resolution for:\n"
                    f"Customer: {self.ticket_data['customer']}\nIssue: {self.ticket_data['subject']} - {self.ticket_data['message']}\n"
                    f"Provide secure identity verification instructions and password reset steps."
                )
                self.resolution = llm.call(messages=[{"role": "user", "content": prompt}]).strip()
                return self.resolution

            @listen("other")
            def handle_other(self):
                self.priority = "medium"
                prompt = (
                    f"You are a Technical Support & Product Specialist. Formulate a helpful response for:\n"
                    f"Customer: {self.ticket_data['customer']}\nIssue: {self.ticket_data['subject']} - {self.ticket_data['message']}\n"
                    f"Provide practical troubleshooting tips and next steps."
                )
                self.resolution = llm.call(messages=[{"role": "user", "content": prompt}]).strip()
                return self.resolution

            @listen(or_(handle_billing, handle_access, handle_other))
            def resolve_ticket(self):
                return {
                    "ticket_id": self.ticket_data["id"],
                    "customer_name": self.ticket_data["customer"],
                    "category": self.category,
                    "priority": self.priority,
                    "resolution": self.resolution
                }

        flow = CustomerSupportFlow()
        flow_res = flow.kickoff()
        return {
            "project": "05 - Customer Support Flow",
            "result": flow_res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 06 - Resume Screening Flow (Router)
# =============================================================================
class ResumeScreeningRequest(BaseModel):
    candidate_name: str = Field(..., example="Devon Vance")
    job_description: str = Field(..., example="Senior Backend Engineer. Must have 4+ years of Python, FastAPI, Docker, and experience building multi-agent AI systems.")
    resume_text: str = Field(..., example="Devon Vance: 5 years experience as Lead Backend Developer. Built distributed microservices in Python, FastAPI, PostgreSQL, and deployed LLM agentic pipelines with CrewAI and Docker on AWS.")

@app.post("/agent6", summary="06 - Resume Screening Flow (Router)")
def run_agent6(request: ResumeScreeningRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 06: Resume Screening Flow
    Flow workflow: load_resumes -> score_resume (0-10) -> check_score router -> shortlist (Summary Crew) / reject (draft polite email) -> save_results.
    """
    try:
        llm = get_llm(x_gemini_api_key)

        class ResumeScreeningFlow(Flow):
            score: int = 0
            decision: str = ""
            details: str = ""

            @start()
            def load_resumes(self):
                return {
                    "name": request.candidate_name,
                    "job": request.job_description,
                    "resume": request.resume_text
                }

            @listen(load_resumes)
            def score_resume(self, data):
                prompt = (
                    f"Evaluate this candidate against the job description.\n"
                    f"Job Description: {data['job']}\n"
                    f"Resume: {data['resume']}\n\n"
                    f"Provide an integer score from 0 to 10 evaluating suitability.\n"
                    f"Respond with ONLY the integer number (e.g. 8)."
                )
                resp = llm.call(messages=[{"role": "user", "content": prompt}]).strip()
                digits = [int(s) for s in resp.split() if s.isdigit()]
                self.score = digits[0] if digits else 7
                return self.score

            @router(score_resume)
            def check_score(self):
                if self.score >= 7:
                    self.decision = "shortlist"
                    return "shortlist"
                else:
                    self.decision = "reject"
                    return "reject"

            @listen("shortlist")
            def shortlist_candidate(self):
                # Summary Crew evaluating strengths and interview questions
                agent = Agent(
                    role="Technical Talent Recruiter",
                    goal="Identify top strengths, qualification highlights, and suggested interview questions.",
                    backstory="You are an expert recruiter for high-growth tech teams.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Candidate {request.candidate_name} scored {self.score}/10.\nResume:\n{request.resume_text}\nJob:\n{request.job_description}\n\nSummarize top 3 core strengths, key skill alignment, and 2 suggested technical interview questions.",
                    expected_output="Shortlist dossier with strengths, alignment notes, and interview questions.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                res = crew.kickoff()
                self.details = res.raw
                return self.details

            @listen("reject")
            def reject_candidate(self):
                prompt = (
                    f"Write a warm, respectful, and encouraging rejection email to candidate {request.candidate_name} for the position:\n"
                    f"{request.job_description}\n"
                    f"Express sincere appreciation for their time and interest."
                )
                self.details = llm.call(messages=[{"role": "user", "content": prompt}]).strip()
                return self.details

            @listen(or_(shortlist_candidate, reject_candidate))
            def save_results(self):
                return {
                    "candidate_name": request.candidate_name,
                    "score": self.score,
                    "decision": self.decision,
                    "details": self.details
                }

        flow = ResumeScreeningFlow()
        res = flow.kickoff()
        return {
            "project": "06 - Resume Screening Flow",
            "result": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 07 - Bug Triage Flow (Router + Crews)
# =============================================================================
class BugIssueRequest(BaseModel):
    issue_title: str = Field(..., example="PostgreSQL connection pool exhausted under concurrent load")
    issue_body: str = Field(..., example="When active API traffic surpasses 400 req/sec, the server starts throwing sqlalchemy.exc.TimeoutError. It looks like connections are not being released back to the pool.")

@app.post("/agent7", summary="07 - Bug Triage Flow (Router + Crews)")
def run_agent7(request: BugIssueRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 07: Bug Triage Flow
    Flow + Crews: read_issue -> classify_issue -> route_type -> Bug Crew / Feature Crew / Answer Crew -> post_reply.
    """
    try:
        llm = get_llm(x_gemini_api_key)

        class BugTriageFlow(Flow):
            issue_type: str = "bug"
            triage_details: str = ""

            @start()
            def read_issue(self):
                return {"title": request.issue_title, "body": request.issue_body}

            @listen(read_issue)
            def classify_issue(self, issue):
                prompt = (
                    f"Classify this issue tracker ticket into EXACTLY ONE category: 'bug', 'feature', or 'question'.\n"
                    f"Title: {issue['title']}\n"
                    f"Body: {issue['body']}\n\n"
                    f"Respond with only the single word: bug, feature, or question."
                )
                resp = llm.call(messages=[{"role": "user", "content": prompt}]).strip().lower()
                if "feature" in resp:
                    self.issue_type = "feature"
                elif "question" in resp:
                    self.issue_type = "question"
                else:
                    self.issue_type = "bug"
                return self.issue_type

            @router(classify_issue)
            def route_type(self):
                return self.issue_type

            @listen("bug")
            def run_bug_crew(self):
                agent = Agent(
                    role="Site Reliability & QA Engineer",
                    goal="Diagnose the bug severity, root cause hypothesis, and reproduction steps.",
                    backstory="You are an expert debugger and reliability specialist.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Analyze this bug report:\nTitle: {request.issue_title}\nBody: {request.issue_body}\n\nDetermine severity (Critical/High/Medium/Low), likely root cause, steps to reproduce, and mitigation steps.",
                    expected_output="Detailed bug triage report with severity and mitigation.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.triage_details = crew.kickoff().raw
                return self.triage_details

            @listen("feature")
            def run_feature_crew(self):
                agent = Agent(
                    role="Technical Product Manager",
                    goal="Convert feature request into a formal user story with acceptance criteria.",
                    backstory="You formulate crisp agile specifications and user stories.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Convert this feature suggestion into an agile story:\nTitle: {request.issue_title}\nBody: {request.issue_body}\n\nFormat as 'As a... I want... So that...', followed by 3 acceptance criteria.",
                    expected_output="Agile user story with acceptance criteria.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.triage_details = crew.kickoff().raw
                return self.triage_details

            @listen("question")
            def run_answer_crew(self):
                agent = Agent(
                    role="Developer Advocate & Documentation Lead",
                    goal="Provide a comprehensive technical answer citing best practices.",
                    backstory="You write developer documentation and assist community engineers.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Answer this technical question with clarity:\nTitle: {request.issue_title}\nBody: {request.issue_body}",
                    expected_output="Comprehensive technical response with code snippets or references.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.triage_details = crew.kickoff().raw
                return self.triage_details

            @listen(or_(run_bug_crew, run_feature_crew, run_answer_crew))
            def post_reply(self):
                return {
                    "issue_title": request.issue_title,
                    "classification": self.issue_type,
                    "triage_details": self.triage_details
                }

        flow = BugTriageFlow()
        res = flow.kickoff()
        return {
            "project": "07 - Bug Triage Flow",
            "result": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 08 - Content Repurposing Flow (and_)
# =============================================================================
class ContentRepurposingRequest(BaseModel):
    article_title: str = Field(..., example="Why Multi-Agent Systems are the Future of Software Engineering")
    article_text: str = Field(
        ...,
        example="In 2026, single-prompt AI is being replaced by multi-agent architectures. In a multi-agent system, specialized agents with distinct backstories, tools, and goals collaborate sequentially or hierarchically to break down complex tasks. This prevents hallucination, introduces automated peer review, and allows deterministic routing via state machines and flows."
    )

@app.post("/agent8", summary="08 - Content Repurposing Flow (and_)")
def run_agent8(request: ContentRepurposingRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 08: Content Repurposing Flow
    Flow + Crews (and_): load_article -> extract_points -> LinkedIn Crew + Twitter Crew + Newsletter Crew in parallel -> publish_pack.
    """
    try:
        llm = get_llm(x_gemini_api_key)

        class ContentRepurposingFlow(Flow):
            points: str = ""
            linkedin_post: str = ""
            twitter_thread: str = ""
            newsletter: str = ""

            @start()
            def load_article(self):
                return {"title": request.article_title, "text": request.article_text}

            @listen(load_article)
            def extract_points(self, article):
                prompt = (
                    f"Extract the 3 core themes, key data points, and big takeaways from this article:\n"
                    f"Title: {article['title']}\n"
                    f"Text: {article['text']}\n\n"
                    f"Provide a structured bulleted summary of key points."
                )
                self.points = llm.call(messages=[{"role": "user", "content": prompt}]).strip()
                return self.points

            @listen(extract_points)
            def generate_linkedin(self):
                agent = Agent(
                    role="LinkedIn Thought Leadership Ghostwriter",
                    goal="Craft a high-engagement 150-word LinkedIn post with a strong hook, insights, and a conversation-starting call to action.",
                    backstory="You write viral B2B leadership content on LinkedIn.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Using these key points:\n{self.points}\n\nWrite a 150-word LinkedIn post with an irresistible first line hook, clear spacing, and 3 hashtags.",
                    expected_output="A polished 150-word LinkedIn post.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.linkedin_post = crew.kickoff().raw
                return self.linkedin_post

            @listen(extract_points)
            def generate_twitter(self):
                agent = Agent(
                    role="X / Twitter Growth Strategist",
                    goal="Convert key insights into a punchy 6-tweet thread.",
                    backstory="You create viral Twitter threads that break down complex technical topics into easy-to-read insights.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Using these key points:\n{self.points}\n\nWrite a 6-tweet thread numbered 1/6 through 6/6. Tweet 1 is the hook, tweets 2-5 are insights, tweet 6 is a conclusion.",
                    expected_output="A 6-tweet numbered thread.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.twitter_thread = crew.kickoff().raw
                return self.twitter_thread

            @listen(extract_points)
            def generate_newsletter(self):
                agent = Agent(
                    role="Tech Newsletter Publisher",
                    goal="Write a concise newsletter edition featuring an intro and 3 highlighted takeaways.",
                    backstory="You publish a widely read weekly technology newsletter.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Using these key points:\n{self.points}\n\nDraft a newsletter section titled '{request.article_title}'. Include a welcoming 2-sentence intro, 3 bulleted takeaways with bold titles, and a sign-off.",
                    expected_output="A complete newsletter edition.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.newsletter = crew.kickoff().raw
                return self.newsletter

            @listen(and_(generate_linkedin, generate_twitter, generate_newsletter))
            def publish_pack(self):
                return {
                    "original_title": request.article_title,
                    "linkedin_post": self.linkedin_post,
                    "twitter_thread": self.twitter_thread,
                    "newsletter": self.newsletter
                }

        flow = ContentRepurposingFlow()
        res = flow.kickoff()
        return {
            "project": "08 - Content Repurposing Flow",
            "result": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 09 - Lead Qualification Flow (Router + Crews)
# =============================================================================
class LeadQualificationRequest(BaseModel):
    lead_name: str = Field(..., example="Starlight Health")
    lead_website: str = Field(..., example="https://starlighthealth.io")
    lead_notes: str = Field(..., example="500-employee healthcare network looking to deploy enterprise multi-agent patient triage within 30 days. Budget approved.")

@app.post("/agent9", summary="09 - Lead Qualification Flow (Router + Crews)")
def run_agent9(request: LeadQualificationRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 09: Lead Qualification Flow
    Flow + Crews (Router): capture_lead -> Enrich Crew -> score_lead router -> hot (Outreach Crew) / warm (nurture) / cold (archive) -> log_lead.
    """
    try:
        llm = get_llm(x_gemini_api_key)

        class LeadQualificationFlow(Flow):
            lead_tier: str = "warm"
            enrichment: str = ""
            action_plan: str = ""

            @start()
            def capture_lead(self):
                return {
                    "name": request.lead_name,
                    "website": request.lead_website,
                    "notes": request.lead_notes
                }

            @listen(capture_lead)
            def run_enrich_crew(self, lead):
                agent = Agent(
                    role="B2B Sales Intelligence Analyst",
                    goal="Evaluate company profile, buying intent signals, and budget fit.",
                    backstory="You research enterprise accounts to determine ICP fit and urgency.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Enrich this lead:\nCompany: {lead['name']}\nWebsite: {lead['website']}\nNotes: {lead['notes']}\n\nAssess firmographic fit, buying intent, and project urgency.",
                    expected_output="An account qualification brief.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.enrichment = crew.kickoff().raw
                return self.enrichment

            @router(run_enrich_crew)
            def score_lead(self):
                prompt = (
                    f"Evaluate this enriched lead and assign EXACTLY ONE tier: 'hot', 'warm', or 'cold'.\n"
                    f"Lead Notes: {request.lead_notes}\n"
                    f"Enrichment: {self.enrichment}\n\n"
                    f"Respond with only the single word: hot, warm, or cold."
                )
                resp = llm.call(messages=[{"role": "user", "content": prompt}]).strip().lower()
                if "hot" in resp:
                    self.lead_tier = "hot"
                elif "cold" in resp:
                    self.lead_tier = "cold"
                else:
                    self.lead_tier = "warm"
                return self.lead_tier

            @listen("hot")
            def run_outreach_crew(self):
                agent = Agent(
                    role="Enterprise Account Executive",
                    goal="Draft a high-priority personalized executive outreach email.",
                    backstory="You close high-value deals with bespoke, compelling communications.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Write a personalized, high-priority email to {request.lead_name} proposing an executive demo this week based on their urgency: {request.lead_notes}",
                    expected_output="An executive email draft ready for immediate dispatch.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.action_plan = crew.kickoff().raw
                return self.action_plan

            @listen("warm")
            def add_to_nurture(self):
                prompt = (
                    f"Draft a helpful educational nurture email for {request.lead_name} sharing relevant case studies and inviting them to an upcoming webinar."
                )
                self.action_plan = llm.call(messages=[{"role": "user", "content": prompt}]).strip()
                return self.action_plan

            @listen("cold")
            def archive_lead(self):
                self.action_plan = f"Lead {request.lead_name} archived. Added to quarterly automated check-in sequence due to low current urgency."
                return self.action_plan

            @listen(or_(run_outreach_crew, add_to_nurture, archive_lead))
            def log_lead(self):
                return {
                    "lead_name": request.lead_name,
                    "tier": self.lead_tier,
                    "enrichment_summary": self.enrichment,
                    "action_plan": self.action_plan
                }

        flow = LeadQualificationFlow()
        res = flow.kickoff()
        return {
            "project": "09 - Lead Qualification Flow",
            "result": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 10 - Approval Report Flow (Human Feedback)
# =============================================================================
class ApprovalReportRequest(BaseModel):
    topic: str = Field(..., example="Post-Quantum Cryptography Migration Strategy")
    audience: str = Field(..., example="Chief Information Security Officers (CISOs)")
    approved: bool = Field(False, example=False, description="Simulate human review: True to approve, False to request revisions")
    feedback: Optional[str] = Field(None, example="Include a specific 3-phase transition timeline for replacing RSA-2048.")

@app.post("/agent10", summary="10 - Approval Report Flow (Human Feedback)")
def run_agent10(request: ApprovalReportRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Project 10: Approval Report Flow
    Flow + Crews (Human feedback loop): collect_topic -> Research Crew -> Writing Crew -> review_draft (human input) -> router approved/revise -> publish or Revision Crew.
    """
    try:
        llm = get_llm(x_gemini_api_key)

        class ApprovalReportFlow(Flow):
            research: str = ""
            draft: str = ""
            final_report: str = ""
            decision: str = "revise"

            @start()
            def collect_topic(self):
                return {"topic": request.topic, "audience": request.audience}

            @listen(collect_topic)
            def run_research_crew(self, data):
                agent = Agent(
                    role="Principal Subject Matter Researcher",
                    goal=f"Research technical facts, risks, and industry standards for: {data['topic']}",
                    backstory="You compile authoritative technical dossiers for executive leadership.",
                    llm=llm,
                    tools=[duckduckgo_search_tool],
                    verbose=False
                )
                task = Task(
                    description=f"Gather 3 essential facts and security recommendations on: {data['topic']} for audience: {data['audience']}.",
                    expected_output="Research briefing document.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.research = crew.kickoff().raw
                return self.research

            @listen(run_research_crew)
            def run_writing_crew(self):
                agent = Agent(
                    role="Executive Report Author",
                    goal=f"Draft an initial report on {request.topic} tailored for {request.audience}.",
                    backstory="You write whitepapers and executive briefs for C-level readers.",
                    llm=llm,
                    verbose=False
                )
                task = Task(
                    description=f"Using this research:\n{self.research}\n\nDraft a structured report for {request.audience} on {request.topic}. Include Executive Summary, Key Findings, and Strategic Imperatives.",
                    expected_output="First draft of the executive report.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.draft = crew.kickoff().raw
                return self.draft

            @listen(run_writing_crew)
            def review_draft(self):
                # Evaluate simulated human review input provided via API request
                if request.approved:
                    self.decision = "approved"
                else:
                    self.decision = "revise"
                return self.decision

            @router(review_draft)
            def check_decision(self):
                return self.decision

            @listen("approved")
            def publish_report(self):
                self.final_report = f"# [PUBLISHED & VERIFIED]\n\n{self.draft}\n\n---\n*Approved for release to {request.audience}*"
                return self.final_report

            @listen("revise")
            def run_revision_crew(self):
                agent = Agent(
                    role="Senior Editorial Reviewer",
                    goal="Incorporate human reviewer feedback and elevate the report draft to perfection.",
                    backstory="You implement critical stakeholder feedback with speed and editorial finesse.",
                    llm=llm,
                    verbose=False
                )
                feedback_str = request.feedback or "Enhance depth, include actionable roadmap, and strengthen conclusion."
                task = Task(
                    description=f"Original Draft:\n{self.draft}\n\nHuman Reviewer Feedback:\n{feedback_str}\n\nRevise and enhance the report to address every item of feedback thoroughly.",
                    expected_output="Final revised and improved executive report.",
                    agent=agent
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                self.final_report = f"# [REVISED BASED ON FEEDBACK]\n\n{crew.kickoff().raw}\n\n---\n*Incorporated feedback: {feedback_str}*"
                return self.final_report

            @listen(or_(publish_report, run_revision_crew))
            def final_response(self):
                return {
                    "topic": request.topic,
                    "audience": request.audience,
                    "review_status": self.decision,
                    "report": self.final_report
                }

        flow = ApprovalReportFlow()
        res = flow.kickoff()
        return {
            "project": "10 - Approval Report Flow",
            "result": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
