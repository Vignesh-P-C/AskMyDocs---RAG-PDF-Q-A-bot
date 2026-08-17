# backend/main.py
import os
import time
import uuid
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import index_document, answer_question

SESSION_TTL_MINUTES = 30

app = FastAPI(title="AskMyDocs API")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions = {}


def _cleanup_expired_sessions():
    cutoff = time.time() - (SESSION_TTL_MINUTES * 60)
    expired = [sid for sid, s in _sessions.items() if s["last_used"] < cutoff]
    for sid in expired:
        del _sessions[sid]


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    _cleanup_expired_sessions()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        retriever, page_count, chunk_count = index_document(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Couldn't index that PDF: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "retriever": retriever,
        "filename": file.filename,
        "last_used": time.time(),
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "page_count": page_count,
        "chunk_count": chunk_count,
    }


@app.post("/ask")
async def ask(req: AskRequest):
    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired — please upload the PDF again.",
        )

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session["last_used"] = time.time()

    try:
        answer = answer_question(session["retriever"], req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong answering that: {e}")

    return {"answer": answer}
