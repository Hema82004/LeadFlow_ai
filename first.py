# leads_api.py
import os
import re
import json
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Literal

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Google Cloud Vision + Vertex AI (Gemini)
from google.cloud import vision
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
SA_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

if not PROJECT_ID:
    raise RuntimeError("GCP_PROJECT_ID env var is required")

# Vertex AI (Gemini)
init(project=PROJECT_ID, location=REGION)
model = GenerativeModel("gemini-2.5-pro")

# Firebase Admin (use service account if provided, else ADC)
if not firebase_admin._apps:
    if SA_PATH and os.path.exists(SA_PATH):
        cred = credentials.Certificate(SA_PATH)
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()
db = firestore.client()

# Vision client (ADC or SA via GOOGLE_APPLICATION_CREDENTIALS)
vision_client = vision.ImageAnnotatorClient()

# FastAPI
app = FastAPI(title="Lead Generation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS
# =========================
PipelineStage = Literal[
    "new", "contacted", "qualified", "proposal", "won", "lost", "nurturing"
]

class LeadTextInput(BaseModel):
    text: str
    user_id: str  # owner or requester
    default_stage: PipelineStage = "new"
    source: Optional[str] = None   # e.g., "webform", "event", "inbound_email"

class LeadImageInput(BaseModel):
    user_id: str
    default_stage: PipelineStage = "new"
    source: Optional[str] = None

class SearchLeadsInput(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    stage: Optional[PipelineStage] = None
    owner_id: Optional[str] = None
    campaign: Optional[str] = None
    min_score: Optional[int] = None
    max_score: Optional[int] = None
    from_date: Optional[str] = None  # YYYY-MM-DD (created_at)
    to_date: Optional[str] = None    # YYYY-MM-DD (created_at)
    limit: int = 50

class UpdateStageInput(BaseModel):
    lead_id: str
    stage: PipelineStage
    owner_id: Optional[str] = None
    next_action: Optional[str] = None
    next_action_date: Optional[str] = None  # YYYY-MM-DD

class ChatLeadsInput(BaseModel):
    user_id: Optional[str] = None
    owner_id: Optional[str] = None
    question: str
    stage: Optional[PipelineStage] = None
    campaign: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None

class VerifyLeadInput(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None

# =========================
# UTILITIES
# =========================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\d{3,5})[-.\s]?\d{3,4}[-.\s]?\d{3,4})")
CURRENCY_RE = re.compile(r"(?:₹|INR|Rs\.?|USD|\$)\s*([0-9][0-9,]*\.?[0-9]{0,2})", re.IGNORECASE)

def friendly_reply(message: str) -> dict:
    return {"reply": message}

def safe_extract_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    m = re.search(r"json\s*(\{.*?\})\s*", raw, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        cleaned = m.group(1).replace("\n", " ").replace("\r", " ")
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

def extract_email_phone(text: str) -> Dict[str, Optional[str]]:
    email = None
    phone = None
    if text:
        e = EMAIL_RE.findall(text)
        p = PHONE_RE.findall(text)
        email = e[0] if e else None
        phone = p[0] if p else None
    return {"email": email, "phone": phone}

def extract_budget(text: str) -> Optional[float]:
    if not text:
        return None
    hits = CURRENCY_RE.findall(text)
    if not hits:
        return None
    try:
        vals = [float(x.replace(",", "")) for x in hits]
        return max(vals) if vals else None
    except Exception:
        return None

def llm_extract_lead_fields(text: str) -> Dict[str, Any]:
    """
    Ask Gemini to extract structured lead details.
    """
    prompt = f"""
You are a sales ops assistant. From the free-form text below, extract a SINGLE JSON object with lead details.
Return ONLY JSON (no comments, no prose).

Required keys (use null if unknown):
- "full_name": string|null
- "email": string|null
- "phone": string|null
- "company": string|null
- "job_title": string|null
- "location": string|null
- "campaign": string|null
- "source": string|null  // e.g., webform, event, referral, linkedin
- "budget": number|null  // numeric if you can infer
- "need_summary": string|null  // short 1-2 lines on pain point / intent
- "notes": string|null
- "tags": [string]       // keywords like 'hot', 'enterprise', 'pilot', 'demo'

Text:
{text}
"""
    resp = model.generate_content(prompt)
    return safe_extract_json(resp.text)

def normalize_lead_fields(d: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        d = {}
    out: Dict[str, Any] = {}

    def pick(*keys, default=None):
        for k in keys:
            v = d.get(k)
            if v not in (None, "", []):
                return v
        return default

    out["full_name"] = pick("full_name", "name")
    out["email"] = pick("email")
    out["phone"] = pick("phone")
    out["company"] = pick("company", "organization")
    out["job_title"] = pick("job_title", "title", "role")
    out["location"] = pick("location", "city")
    out["campaign"] = pick("campaign")
    out["source"] = pick("source", default=defaults.get("source"))
    # budget
    budget = pick("budget")
    if isinstance(budget, str):
        try:
            budget = float(re.sub(r"[^\d.]", "", budget))
        except Exception:
            budget = None
    out["budget"] = budget
    out["need_summary"] = pick("need_summary", "intent", "summary")
    out["notes"] = pick("notes")
    tags = pick("tags")
    if isinstance(tags, list):
        out["tags"] = [str(t).strip() for t in tags if str(t).strip()]
    elif isinstance(tags, str):
        out["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        out["tags"] = []
    # defaults
    out["stage"] = defaults.get("stage", "new")
    out["owner_id"] = defaults.get("owner_id") or defaults.get("user_id")
    out["score"] = defaults.get("score", None)
    # backfill email/phone/budget via regex if missing
    return out

def enrich_missing_with_regex(out: Dict[str, Any], text: str) -> Dict[str, Any]:
    if not out.get("email") or not out.get("phone"):
        m = extract_email_phone(text)
        out["email"] = out.get("email") or m["email"]
        out["phone"] = out.get("phone") or m["phone"]
    if out.get("budget") in (None, "", "N/A"):
        b = extract_budget(text)
        if b is not None:
            out["budget"] = b
    return out

def basic_score(lead: Dict[str, Any]) -> int:
    """
    Simple, rule-based score 0-100 you can tweak later or replace with LLM.
    """
    score = 10
    if lead.get("email"): score += 15
    if lead.get("phone"): score += 10
    if lead.get("company"): score += 10
    if lead.get("job_title"): score += 10
    if lead.get("budget"): score += 15
    if lead.get("need_summary"): score += 15
    tags = set([t.lower() for t in (lead.get("tags") or [])])
    if {"hot", "urgent", "pilot"}.intersection(tags): score += 10
    return max(0, min(100, score))

def store_lead(payload: dict) -> str:
    clean = {k: v for k, v in payload.items() if v not in [None, ""]}
    clean["created_at"] = datetime.utcnow().isoformat()
    clean["last_contact_date"] = clean.get("last_contact_date") or None
    ref = db.collection("leads").add(clean)
    return ref[1].id  # document id

def retrieve_leads(filters: dict = None, limit: int = 50) -> List[Dict[str, Any]]:
    col = db.collection("leads")
    if filters:
        for k, v in filters.items():
            if v not in (None, ""):
                col = col.where(k, "==", v)
    docs = col.order_by("created_at").limit(limit).stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]

def summarize_leads(leads: list) -> str:
    if not leads:
        return "No leads found."
    lines = ["Here are your recent leads:"]
    for l in leads:
        name = l.get("full_name") or "Unknown"
        comp = l.get("company") or "-"
        stage = l.get("stage") or "new"
        score = l.get("score")
        created = l.get("created_at", "")[:10]
        score_disp = f"{score}" if isinstance(score, int) else "N/A"
        lines.append(f"- *{name}* @ {comp} — stage: {stage}, score: {score_disp}, created: {created}")
    lines.append("\nNeed more filters or details?")
    return "\n".join(lines)

# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return friendly_reply("👋 Lead Generation API is live.")

# Store lead from free-form text (email body, webform blob, chat transcript, etc.)
@app.post("/store_lead_text")
def store_lead_text(inp: LeadTextInput):
    extracted = llm_extract_lead_fields(inp.text)
    lead = normalize_lead_fields(extracted, {"stage": inp.default_stage, "user_id": inp.user_id, "source": inp.source})
    lead = enrich_missing_with_regex(lead, inp.text)
    # compute score if not provided
    if lead.get("score") is None:
        lead["score"] = basic_score(lead)
    lead["owner_id"] = lead.get("owner_id") or inp.user_id
    lead_id = store_lead(lead)
    return friendly_reply(f"Saved lead ({lead_id}): {lead.get('full_name') or lead.get('email') or 'Unnamed lead'}.")

# Store lead from an image (Vision OCR -> Gemini -> Firestore)
@app.post("/store_lead_image")
async def store_lead_image(user_id: str = Query(...), default_stage: PipelineStage = Query("new"), source: Optional[str] = Query(None), file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload")

    image = vision.Image(content=image_bytes)
    ocr_result = vision_client.text_detection(image=image)
    ocr_text = ocr_result.text_annotations[0].description if ocr_result.text_annotations else ""
    if not ocr_text.strip():
        return friendly_reply("I couldn’t read any text from that image. Try a clearer image?")

    extracted = llm_extract_lead_fields(ocr_text)
    lead = normalize_lead_fields(extracted, {"stage": default_stage, "user_id": user_id, "source": source})
    lead = enrich_missing_with_regex(lead, ocr_text)
    if lead.get("score") is None:
        lead["score"] = basic_score(lead)
    lead["owner_id"] = lead.get("owner_id") or user_id
    lead_id = store_lead(lead)
    return friendly_reply(f"Lead captured from image ({lead_id}).")

# List leads (simple summary)
@app.get("/get_leads")
def get_leads(user_id: Optional[str] = None, limit: int = 50):
    filters = {"owner_id": user_id} if user_id else None
    leads = retrieve_leads(filters, limit=limit)
    return friendly_reply(summarize_leads(leads))

# Search / filter leads (POST body for richer filters)
@app.post("/leads/search")
def search_leads(q: SearchLeadsInput):
    # Firestore composite filtering: apply equality filters in query, others in-memory
    eq_filters = {}
    if q.user_id: eq_filters["owner_id"] = q.user_id
    if q.email: eq_filters["email"] = q.email
    if q.phone: eq_filters["phone"] = q.phone
    if q.company: eq_filters["company"] = q.company
    if q.stage: eq_filters["stage"] = q.stage
    if q.campaign: eq_filters["campaign"] = q.campaign

    candidates = retrieve_leads(eq_filters, limit=max(200, q.limit))
    # in-memory range filters
    def in_range(ld):
        ok = True
        if q.min_score is not None:
            ok = ok and isinstance(ld.get("score"), int) and ld["score"] >= q.min_score
        if q.max_score is not None:
            ok = ok and isinstance(ld.get("score"), int) and ld["score"] <= q.max_score
        if q.from_date or q.to_date:
            lo = q.from_date or "0000-01-01"
            hi = q.to_date or "9999-12-31"
            created = (ld.get("created_at") or "")[:10]
            ok = ok and (created and lo <= created <= hi)
        return ok

    results = [ld for ld in candidates if in_range(ld)]
    return {"count": len(results[:q.limit]), "leads": results[:q.limit]}

# Update a lead's stage / ownership / next action
@app.post("/leads/update_stage")
def update_stage(body: UpdateStageInput):
    ref = db.collection("leads").document(body.lead_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Lead not found")
    updates = {"stage": body.stage, "updated_at": datetime.utcnow().isoformat()}
    if body.owner_id: updates["owner_id"] = body.owner_id
    if body.next_action: updates["next_action"] = body.next_action
    if body.next_action_date:
        # store ISO date; validate roughly
        try:
            date.fromisoformat(body.next_action_date)
            updates["next_action_date"] = body.next_action_date
        except Exception:
            updates["next_action_date"] = None
    ref.update(updates)
    return friendly_reply(f"Lead {body.lead_id} moved to stage '{body.stage}'.")

# Free-form Q&A over leads
@app.post("/chat_leads")
def chat_leads(body: ChatLeadsInput):
    # build query
    col = db.collection("leads")
    if body.user_id:
        col = col.where("owner_id", "==", body.user_id)
    if body.owner_id:
        col = col.where("owner_id", "==", body.owner_id)
    if body.stage:
        col = col.where("stage", "==", body.stage)
    if body.campaign:
        col = col.where("campaign", "==", body.campaign)

    docs = list(col.stream())
    leads = [d.to_dict() | {"id": d.id} for d in docs]

    if body.from_date or body.to_date:
        lo = body.from_date or "0000-01-01"
        hi = body.to_date or "9999-12-31"
        def in_window(ld):
            created = (ld.get("created_at") or "")[:10]
            return created and lo <= created <= hi
        leads = [ld for ld in leads if in_window(ld)]

    if not leads:
        return friendly_reply("I don’t see any leads matching those filters. Try loosening them?")

    leads_json = json.dumps(leads, ensure_ascii=False)
    prompt = f"""
You are a helpful sales assistant. Answer the user's question about their leads.

User question: {body.question}

Leads JSON:
{leads_json}

Guidelines:
- Be concise and factual. Do NOT invent values.
- If aggregations are requested (counts, totals, averages), compute them from the JSON and show the result.
- When referring to specific leads, mention full_name, company, stage, and created_at date.
- End with a short follow-up like: "Anything else you want to check?"
"""
    resp = model.generate_content(prompt)
    answer = (resp.text or "").strip() or "Sorry, I couldn’t generate a response."
    return friendly_reply(answer)

# Verify lead existence (basic de-dup)
@app.post("/leads/verify")
def verify_lead(q: VerifyLeadInput):
    col = db.collection("leads").limit(100)
    candidates = [{"id": d.id, **d.to_dict()} for d in col.stream()]

    def matches(ld):
        ok = True
        if q.email:
            ok = ok and (ld.get("email") or "").lower() == q.email.lower()
        if q.phone:
            ok = ok and (re.sub(r"\D", "", ld.get("phone") or "") == re.sub(r"\D", "", q.phone))
        if q.company:
            ok = ok and (q.company.lower() in (ld.get("company") or "").lower())
        return ok

    hits = [ld for ld in candidates if matches(ld)]
    return {"count": len(hits), "matches": hits}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run("first:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
