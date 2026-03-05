import os
from datetime import datetime
from uuid import uuid4
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables from .env
load_dotenv()

# Basic safety check
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

app = FastAPI(title="Company Research Report Server", version="0.2.0")

# In-memory storage for MVP
REPORT_STORE: dict[str, dict] = {}

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ── Pydantic model for incoming bundle ────────────────────────────────────────

class RunRequest(BaseModel):
    company_name: str
    ticker: str | None = None
    timestamp: str | None = None
    primary_sources: list[dict] = []
    news: list[dict] = []
    financials: dict = {}


# ── LangGraph state ───────────────────────────────────────────────────────────

class ReportState(TypedDict):
    company_name: str
    ticker: str
    primary_sources: list[dict]
    news: list[dict]
    financials: dict
    executive_summary: str
    company_overview: str
    financial_snapshot: str
    recent_news: str
    market_position: str
    strategic_assessment: str


# ── Helper: format sources for prompt ─────────────────────────────────────────

def format_sources(sources: list[dict], limit: int = 8) -> str:
    # Sort by score descending, then take top results
    sorted_sources = sorted(sources, key=lambda x: x.get("score", 0), reverse=True)
    lines = []
    for s in sorted_sources[:limit]:
        title = s.get("title") or "(no title)"
        snippet = s.get("snippet") or ""
        url = s.get("url") or ""
        doc_type = s.get("doc_type") or "other"
        year = s.get("year_detected") or ""
        year_str = f" [{year}]" if year else ""
        lines.append(f"- [{doc_type.upper()}{year_str}] {title}: {snippet} ({url})")
    return "\n".join(lines) if lines else "No sources available."


def format_news(news: list[dict], limit: int = 8) -> str:
    lines = []
    for n in news[:limit]:
        title = n.get("title") or "(no title)"
        date = n.get("date") or ""
        # Guardian uses 'summary', fallback to 'snippet'
        summary = n.get("summary") or n.get("snippet") or ""
        byline = n.get("byline") or ""
        source = n.get("source") or ""
        meta = " | ".join(filter(None, [source, byline]))
        date_short = date[:10] if date else ""
        lines.append(f"- [{date_short}] {title}: {summary} ({meta})")
    return "\n".join(lines) if lines else "No news available."


def format_financials(financials: dict) -> str:
    if not financials:
        return "No financial data available."

    # Check if Alpha Vantage hit rate limit
    all_values = []
    for section_key, section_val in financials.items():
        if isinstance(section_val, dict):
            # Check for rate limit / info messages
            if "Information" in section_val or "Note" in section_val:
                continue
            for key, value in section_val.items():
                if isinstance(value, (str, int, float)):
                    all_values.append(f"{section_key} — {key}: {value}")
                elif isinstance(value, list) and len(value) > 0:
                    # e.g. annualReports list — take first item
                    first = value[0]
                    if isinstance(first, dict):
                        for k, v in list(first.items())[:10]:
                            all_values.append(f"{section_key} — {k}: {v}")

    if not all_values:
        return "Financial data unavailable (API rate limit reached). Base analysis on sources and news."

    return "\n".join(all_values[:40])


# ── LangGraph nodes ───────────────────────────────────────────────────────────

def node_executive_summary(state: ReportState) -> ReportState:
    prompt = f"""You are a senior business analyst. Write a concise Executive Summary (3-4 sentences) for {state['company_name']} ({state['ticker']}).

Use this research:
{format_sources(state['primary_sources'])}

Recent news:
{format_news(state['news'])}

Financials:
{format_financials(state['financials'])}

Write only the Executive Summary text. No headers, no markdown."""

    result = llm.invoke(prompt)
    state["executive_summary"] = result.content.strip()
    return state


def node_company_overview(state: ReportState) -> ReportState:
    prompt = f"""You are a senior business analyst. Write a Company Overview (3-5 sentences) for {state['company_name']} ({state['ticker']}).
Cover: what the company does, its industry, key products or services, and its market.

Use this research:
{format_sources(state['primary_sources'])}

Write only the Company Overview text. No headers, no markdown."""

    result = llm.invoke(prompt)
    state["company_overview"] = result.content.strip()
    return state


def node_financial_snapshot(state: ReportState) -> ReportState:
    prompt = f"""You are a financial analyst. Write a Financial Snapshot (3-5 sentences) for {state['company_name']} ({state['ticker']}).
Summarize the key financial metrics in plain English. Highlight notable figures.

Financial data:
{format_financials(state['financials'])}

Write only the Financial Snapshot text. No headers, no markdown."""

    result = llm.invoke(prompt)
    state["financial_snapshot"] = result.content.strip()
    return state


def node_recent_news(state: ReportState) -> ReportState:
    prompt = f"""You are a business journalist. Write a Recent News summary (3-5 sentences) for {state['company_name']} ({state['ticker']}).
Summarize the most important recent developments.

News items:
{format_news(state['news'])}

Write only the Recent News summary. No headers, no markdown."""

    result = llm.invoke(prompt)
    state["recent_news"] = result.content.strip()
    return state


def node_market_position(state: ReportState) -> ReportState:
    prompt = f"""You are a market analyst. Write a Market Position analysis (3-5 sentences) for {state['company_name']} ({state['ticker']}).
Cover: competitive standing, key competitors, market share or leadership signals.

Research:
{format_sources(state['primary_sources'])}

News context:
{format_news(state['news'])}

Write only the Market Position text. No headers, no markdown."""

    result = llm.invoke(prompt)
    state["market_position"] = result.content.strip()
    return state


def node_strategic_assessment(state: ReportState) -> ReportState:
    prompt = f"""You are a strategy consultant. Write a Strategic Assessment (3-5 sentences) for {state['company_name']} ({state['ticker']}).
Cover: key opportunities, risks, and strategic priorities based on available information.

Research:
{format_sources(state['primary_sources'])}

News:
{format_news(state['news'])}

Financials:
{format_financials(state['financials'])}

Write only the Strategic Assessment text. No headers, no markdown."""

    result = llm.invoke(prompt)
    state["strategic_assessment"] = result.content.strip()
    return state


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_report_graph():
    graph = StateGraph(ReportState)

    graph.add_node("executive_summary", node_executive_summary)
    graph.add_node("company_overview", node_company_overview)
    graph.add_node("financial_snapshot", node_financial_snapshot)
    graph.add_node("recent_news", node_recent_news)
    graph.add_node("market_position", node_market_position)
    graph.add_node("strategic_assessment", node_strategic_assessment)

    graph.set_entry_point("executive_summary")
    graph.add_edge("executive_summary", "company_overview")
    graph.add_edge("company_overview", "financial_snapshot")
    graph.add_edge("financial_snapshot", "recent_news")
    graph.add_edge("recent_news", "market_position")
    graph.add_edge("market_position", "strategic_assessment")
    graph.add_edge("strategic_assessment", END)

    return graph.compile()


report_graph = build_report_graph()


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html_report(state: ReportState, report_id: str, created_at: str) -> str:
    sources_html = "".join([
        f"<li><a href='{s.get('url') or s.get('link', '')}' target='_blank'>{s.get('title') or s.get('name', '(no title)')}</a></li>"
        for s in state["primary_sources"][:8]
    ])
    news_html = "".join([
        f"<li><span class='date'>{(n.get('date') or '')[:10]}</span> — "
        f"<a href='{n.get('url', '')}' target='_blank'>{n.get('title', '(no title)')}</a>"
        f"<br><span class='summary'>{n.get('summary') or n.get('snippet') or ''}</span></li>"
        for n in state["news"][:8]
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Intelligence Report: {state['company_name']}</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; background: #f9f9f9; }}
    .header {{ background: #0f172a; color: white; padding: 32px; border-radius: 8px; margin-bottom: 32px; }}
    .header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    .header .meta {{ color: #94a3b8; font-size: 14px; }}
    .section {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .section h2 {{ margin: 0 0 12px 0; font-size: 18px; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
    .section p {{ line-height: 1.7; color: #374151; margin: 0; }}
    .section ul {{ margin: 0; padding-left: 20px; }}
    .section li {{ margin-bottom: 6px; line-height: 1.5; color: #374151; }}
    .section a {{ color: #2563eb; text-decoration: none; }}
    .section a:hover {{ text-decoration: underline; }}
    .date {{ color: #6b7280; font-size: 13px; }}
    .summary {{ color: #6b7280; font-size: 13px; font-style: italic; }}
    .footer {{ text-align: center; color: #9ca3af; font-size: 12px; margin-top: 32px; padding-bottom: 40px; }}
    .badge {{ display: inline-block; background: #dbeafe; color: #1d4ed8; padding: 2px 10px; border-radius: 99px; font-size: 13px; font-weight: 600; margin-left: 8px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{state['company_name']} <span class="badge">{state['ticker']}</span></h1>
    <div class="meta">Market Intelligence Report &nbsp;·&nbsp; Generated {created_at} &nbsp;·&nbsp; Report ID: {report_id}</div>
  </div>

  <div class="section">
    <h2>Executive Summary</h2>
    <p>{state['executive_summary']}</p>
  </div>

  <div class="section">
    <h2>Company Overview</h2>
    <p>{state['company_overview']}</p>
  </div>

  <div class="section">
    <h2>Financial Snapshot</h2>
    <p>{state['financial_snapshot']}</p>
  </div>

  <div class="section">
    <h2>Recent News</h2>
    <p>{state['recent_news']}</p>
    <br>
    <ul>{news_html}</ul>
  </div>

  <div class="section">
    <h2>Market Position</h2>
    <p>{state['market_position']}</p>
  </div>

  <div class="section">
    <h2>Strategic Assessment</h2>
    <p>{state['strategic_assessment']}</p>
  </div>

  <div class="section">
    <h2>Sources</h2>
    <ul>{sources_html}</ul>
  </div>

  <div class="footer">Generated by Company Research Agent &nbsp;·&nbsp; Powered by LangGraph + GPT-4o-mini</div>
</body>
</html>"""


# ── FastAPI routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/run")
def run_report(req: RunRequest):
    report_id = str(uuid4())
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build initial state
    initial_state: ReportState = {
        "company_name": req.company_name,
        "ticker": req.ticker or "N/A",
        "primary_sources": req.primary_sources,
        "news": req.news,
        "financials": req.financials,
        "executive_summary": "",
        "company_overview": "",
        "financial_snapshot": "",
        "recent_news": "",
        "market_position": "",
        "strategic_assessment": "",
    }

    # Run LangGraph
    final_state = report_graph.invoke(initial_state)

    # Build HTML
    html = build_html_report(final_state, report_id, created_at)

    REPORT_STORE[report_id] = {
        "report_id": report_id,
        "company_name": req.company_name,
        "created_at": created_at,
        "html": html,
    }

    return {
        "report_id": report_id,
        "report_url": f"/reports/{report_id}",
        "download_url": f"/reports/{report_id}/download",
    }


@app.get("/reports/{report_id}", response_class=HTMLResponse)
def view_report(report_id: str):
    report = REPORT_STORE.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return HTMLResponse(content=report["html"])


@app.get("/reports/{report_id}/download")
def download_report(report_id: str):
    report = REPORT_STORE.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    from fastapi.responses import Response
    return Response(
        content=report["html"],
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename=report_{report_id[:8]}.html"}
    )