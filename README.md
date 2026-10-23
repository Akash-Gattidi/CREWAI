# CREWAI - 10 Autonomous Multi-Agent Systems & Flows

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![CrewAI](https://img.shields.io/badge/CrewAI-1.15+-FF5A50.svg?style=flat)](https://crewai.com)
[![Gemini](https://img.shields.io/badge/Gemini_3.1_Flash_Lite_Preview-4285F4.svg?style=flat&logo=google)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade FastAPI application implementing all 10 CrewAI multi-agent crews and flows described in `crewai_projects.html`, powered universally by Google's **Gemini 3.1 Flash Lite Preview** (`gemini/gemini-3.1-flash-lite-preview`).

---

## Architecture Overview

| Route | Project Name | Pattern | Architecture | Description |
|---|---|---|---|---|
| `POST /agent1` | **Blog Post Crew** | Sequential Crew | 3 Agents | Researcher (DuckDuckGo Search) -> Writer (600-word draft) -> Editor (polishes tone/facts). |
| `POST /agent2` | **Competitor Analysis Crew** | Hierarchical Crew | Manager + 4 Agents | Manager assigns tasks in parallel to Site Scraper, Pricing Analyst, News Analyst, and Strategist to produce a SWOT table. |
| `POST /agent3` | **Meeting Notes Crew** | Parallel Tasks | 4 Agents | Summarizer, Action Finder, and Decision Logger run in parallel (`async_execution=True`); Formatter merges them into structured notes. |
| `POST /agent4` | **Data Report Crew** | Structured Output | 3 Agents | Data Analyst extracts metrics (`output_pydantic=SalesStats`) -> Insight Writer (top 3 findings) -> Reviewer (verified report). |
| `POST /agent5` | **Customer Support Flow** | Router Flow | Flow + State | `receive_ticket` -> `categorize_ticket` -> `@router` routes to `billing`, `account_access`, or `other` handler -> `resolve_ticket`. |
| `POST /agent6` | **Resume Screening Flow** | Router Flow | Flow + Crew | `load_resumes` -> `score_resume` (0-10) -> `@router` checks score (>= 7 shortlists with Summary Crew, < 7 drafts polite rejection email). |
| `POST /agent7` | **Bug Triage Flow** | Router Flow + Crews | Flow + 3 Crews | `read_issue` -> `classify_issue` -> `@router` triggers Bug Crew (severity/repro), Feature Crew (user story), or Answer Crew (technical reply). |
| `POST /agent8` | **Content Repurposing Flow** | `and_` Flow + Crews | Flow + 3 Crews | `load_article` -> `extract_points` -> LinkedIn Crew (150 words), Twitter Crew (6 tweets), and Newsletter Crew run concurrently -> `publish_pack`. |
| `POST /agent9` | **Lead Qualification Flow** | Router Flow + Crews | Flow + 2 Crews | `capture_lead` -> Enrich Crew -> `@router` scores lead (`hot`, `warm`, `cold`) -> Outreach Crew (hot) or Nurture (warm) or Archive (cold) -> `log_lead`. |
| `POST /agent10` | **Approval Report Flow** | Human Feedback Loop | Flow + 3 Crews | `collect_topic` -> Research Crew -> Writing Crew -> `review_draft` (simulated human review) -> `@router` approves or sends to Revision Crew. |

---

## Deployment (Render / Vercel / Netlify)

### 1. Deploying on Render (Recommended)
1. Fork or push this repository to GitHub: `https://github.com/Akash-Gattidi/CREWAI.git`
2. Log into [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
3. Select your GitHub repository **CREWAI**.
4. Configure the service settings:
   - **Name**: `crewai-fastapi-service`
   - **Environment**: `Python 3` (Python 3.11 or 3.12 recommended)
   - **Region**: Closest to your users (e.g., Singapore, Frankfurt, Oregon)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: Your Gemini API key.
6. Click **Create Web Service**. Once deployed, Render will provide a live URL like `https://crewai-fastapi-service.onrender.com`.

---

## Local Setup & Development

### Prerequisites
- Python 3.10 - 3.12
- UV package manager (or standard `pip` / `venv`)

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Akash-Gattidi/CREWAI.git
cd CREWAI
uv venv --python 3.12 .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On macOS/Linux
```

### 2. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000
```

### 4. Run the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at:
`http://localhost:8000/docs`

---

## Postman & cURL Testing Guide

Import the following requests into Postman or execute via cURL:

### 1. Agent 1: Blog Post Crew
```bash
curl -X POST "http://localhost:8000/agent1" \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI in sustainable agriculture"}'
```

### 2. Agent 2: Competitor Analysis Crew
```bash
curl -X POST "http://localhost:8000/agent2" \
  -H "Content-Type: application/json" \
  -d '{"competitors": ["Stripe", "Adyen", "PayPal"]}'
```

### 3. Agent 3: Meeting Notes Crew
```bash
curl -X POST "http://localhost:8000/agent3" \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Alex: We need the authentication flow completed by Friday. Sarah: I will finish JWT token refresh and tests by Thursday. John: I will wire the login page UI by Friday noon. Alex: Great, and we all agreed to deprecate legacy auth next sprint."}'
```

### 4. Agent 4: Data Report Crew
```bash
curl -X POST "http://localhost:8000/agent4" \
  -H "Content-Type: application/json" \
  -d '{"csv_data": "product,region,units_sold,revenue\nWidget Alpha,North,120,2400\nWidget Beta,South,80,3200\nWidget Gamma,East,150,1500\nWidget Delta,West,200,6000"}'
```

### 5. Agent 5: Customer Support Flow
```bash
curl -X POST "http://localhost:8000/agent5" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TCK-4819",
    "customer_name": "Alice Johnson",
    "subject": "Charged twice for monthly subscription",
    "message": "I see two identical charges on my Mastercard statement on October 22nd."
  }'
```

### 6. Agent 6: Resume Screening Flow
```bash
curl -X POST "http://localhost:8000/agent6" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_name": "Elena Rostova",
    "job_description": "Senior Backend AI Engineer with 4+ years of FastAPI, CrewAI, and LangChain experience.",
    "resume_text": "Elena Rostova: 6 years building distributed Python backends. Expert in FastAPI, PostgreSQL, LangChain, and orchestrating multi-agent systems with CrewAI on AWS."
  }'
```

### 7. Agent 7: Bug Triage Flow
```bash
curl -X POST "http://localhost:8000/agent7" \
  -H "Content-Type: application/json" \
  -d '{
    "issue_title": "Database connection timeout under heavy concurrent load",
    "issue_body": "At 600 req/sec, the database connection pool exhausts and throws ConnectionTimeoutException. Idle connections are not recycled."
  }'
```

### 8. Agent 8: Content Repurposing Flow
```bash
curl -X POST "http://localhost:8000/agent8" \
  -H "Content-Type: application/json" \
  -d '{
    "article_title": "The Era of Agentic Workflows",
    "article_text": "In 2026, autonomous agentic workflows have surpassed single-shot LLM prompts. By chaining specialized agents with distinct backstories, tools, and validation loops, enterprises are automating mission-critical business processes without hallucinations."
  }'
```

### 9. Agent 9: Lead Qualification Flow
```bash
curl -X POST "http://localhost:8000/agent9" \
  -H "Content-Type: application/json" \
  -d '{
    "lead_name": "Apex Health Systems",
    "lead_website": "https://apexhealth.io",
    "lead_notes": "600-hospital network looking to automate patient discharge notes with multi-agent pipelines. Budget approved for Q4."
  }'
```

### 10. Agent 10: Approval Report Flow
```bash
curl -X POST "http://localhost:8000/agent10" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Post-Quantum Cryptography Migration",
    "audience": "Chief Information Security Officers",
    "approved": true,
    "feedback": "Include transition timeline for NIST algorithms."
  }'
```

---

## License
MIT License. Developed for Advanced Agentic AI Training.
