"""
main.py - Backend API (Role 8)

Ties together: parser -> chunker -> embedder -> vector_store -> retriever -> llm.

NOTE: This is a starter version drafted by Member 7 (LLM Integration) to unblock
testing while Member 8 (Backend API) is still building theirs out. Sync with
Member 8 before merging -- they may already have a different version, and
this should be treated as a proposal, not the final file.

Endpoints:
  POST /upload  -> upload a syllabus PDF, parse/chunk/embed/store it
  POST /query   -> ask a question about the currently uploaded syllabus
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parser import parse_pdf, PDFParsingError       # Member 1
from chunker import chunk_text                     # Member 3 -- confirm actual function name
from embedder import Embedder                      # Member 4
from vector_store import VectorStore                # Member 5
from retriever import SyllabusRetriever             # Member 6
from llm import ask_llm, chunks_to_strings          # Member 7 (you)

app = FastAPI(title="Syllabus Assistant API")

# Allow the frontend (running on a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory session state (single-document scope per PRD) ---------------
# NOTE: per PRD this is intentionally single-session / single-document, no DB.
_state = {
    "embedder": Embedder(),
    "vector_store": None,
    "retriever": None,
    "syllabus_uploaded": False,
}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/upload")
async def upload_syllabus(file: UploadFile = File(...)):
    """
    Accepts a PDF syllabus, extracts text, chunks it, embeds the chunks,
    and builds a fresh vector store + retriever for this session.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in v1.")

    file_bytes = await file.read()

    # --- Member 1's parser: raises PDFParsingError on empty/scanned/corrupt PDFs ---
    try:
        parsed_doc = parse_pdf(file_bytes)
    except PDFParsingError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Member 3's chunking function -- confirm actual signature ---
    # parsed_doc.full_text is the simple option; parsed_doc.pages (list of
    # Page(page_number, text)) is available if chunker wants per-page metadata
    # so retriever/llm can cite exact page numbers per FR #7.
    chunks = chunk_text(parsed_doc.full_text)  # expected: list[str] or list[dict] with metadata

    # --- Member 4's embedder + Member 5's vector store ---
    embedder = _state["embedder"]
    vector_store = VectorStore()
    vector_store.build(chunks, embedder)  # confirm actual method name/signature with Member 5

    _state["vector_store"] = vector_store
    _state["retriever"] = SyllabusRetriever(embedder, vector_store)
    _state["syllabus_uploaded"] = True

    return {"status": "ok", "chunks_indexed": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query_syllabus(request: QueryRequest):
    """
    Answers a question about the currently uploaded syllabus using
    retrieval (Member 6) + grounded generation (Member 7 / this file's llm.py).
    """
    if not _state["syllabus_uploaded"] or _state["retriever"] is None:
        raise HTTPException(status_code=400, detail="No syllabus uploaded yet. Call /upload first.")

    retriever: SyllabusRetriever = _state["retriever"]

    # Get raw chunk dicts (for sources) rather than the pre-joined string,
    # since ask_llm() expects list[str] and we also want metadata for citations.
    retrieved_chunks = retriever.retrieve(request.question, top_k=request.top_k)

    if not retrieved_chunks:
        return QueryResponse(answer="I couldn't find that in the syllabus.", sources=[])

    context_strings = chunks_to_strings(retrieved_chunks)
    answer = ask_llm(request.question, context_strings)

    return QueryResponse(answer=answer, sources=retrieved_chunks)


@app.get("/health")
async def health_check():
    return {"status": "ok", "syllabus_uploaded": _state["syllabus_uploaded"]}
