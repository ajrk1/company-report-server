import os
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables from .env
load_dotenv()

# Basic safety check
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

app = FastAPI(title="Company Research Report Server", version="0.1.0")

# In-memory storage for MVP (we will replace with persistent storage later)
REPORT_STORE: dict[str, dict] = {}


class RunRequest(BaseModel):
    company_name: str
    ticker: str | None = None
    timestamp: str | None = None
    primary_sources: list[dict] = []
    news: list[dict] = []
    financials: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/run")
def run_report(req: RunRequest):
    """
    MVP: accept the combined bundle from n8n and generate a simple HTML report.
    Next step: replace the report generation with LangGraph logic.
    """
    report_id = str(uuid4())

    # Very basic HTML (MVP)
    html = f"""
    <html>
      <head><title>Company Report: {req.company_name}</title></head>
      <body>
        <h1>Company Intelligence Report</h1>

        <h2>Company</h2>
        <p><b>Name:</b> {req.company_name}</p>
        <p><b>Ticker:</b> {req.ticker or ""}</p>
        <p><b>Timestamp:</b> {req.timestamp or ""}</p>

        <h2>Primary Sources</h2>
        <ul>
          {''.join([f"<li><a href='{s.get('url','')}'>{s.get('title','(no title)')}</a></li>" for s in req.primary_sources])}
        </ul>

        <h2>Recent News</h2>
        <ul>
          {''.join([f"<li><a href='{n.get('url','')}'>{n.get('title','(no title)')}</a> — {n.get('date','')}</li>" for n in req.news])}
        </ul>

        <h2>Financials (Raw)</h2>
        <pre>{req.financials}</pre>

      </body>
    </html>
    """.strip()

    REPORT_STORE[report_id] = {
        "report_id": report_id,
        "company_name": req.company_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "html": html,
    }

    return {
        "report_id": report_id,
        "report_url": f"/reports/{report_id}",
        "download_url": f"/reports/{report_id}/download",
    }


@app.get("/reports/{report_id}")
def view_report(report_id: str):
    report = REPORT_STORE.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report["html"]


@app.get("/reports/{report_id}/download")
def download_report(report_id: str):
    report = REPORT_STORE.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Return HTML directly (browser can save it)
    return report["html"]