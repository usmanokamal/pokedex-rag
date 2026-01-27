from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llm import GeminiClient
from rag import FilterSpec, PokemonRAG

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Pokemon RAG Bot", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

rag = PokemonRAG()
llm = GeminiClient()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=25)


class ChatResponse(BaseModel):
    answer: str
    results: List[Dict[str, Any]]
    meta: Dict[str, Any]
    used_llm: bool


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    results, meta = rag.search(request.query, request.top_k)

    context = rag.build_context(results)

    used_llm = False
    if llm.available() and context:
        answer = llm.generate(request.query, context)
        used_llm = True
    else:
        if not results:
            answer = "No matching Pokemon found. Try adjusting your filters or asking differently."
        else:
            answer = rag.format_results(results)

    meta_payload = _serialize_meta(meta)

    return ChatResponse(
        answer=answer,
        results=results,
        meta=meta_payload,
        used_llm=used_llm,
    )


def _serialize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(meta)
    filters = payload.get("filters", [])
    serialized = []
    for filt in filters:
        if isinstance(filt, FilterSpec):
            serialized.append({"column": filt.column, "op": filt.op, "value": filt.value})
        else:
            serialized.append(filt)
    payload["filters"] = serialized
    return payload
