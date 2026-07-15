import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import QueryRequest, QueryResponse
from app.services import rag_service
from app.services.auth_service import User, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    user: User = Depends(get_current_user),
):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    try:
        answer, sources = rag_service.answer_question(
            question, user.id, request.history
        )
        return QueryResponse(answer=answer, sources=sources)
    except Exception as exc:
        logger.exception("Query failed for user %s", user.id)
        raise HTTPException(500, f"Query failed: {exc}") from exc


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    user: User = Depends(get_current_user),
):
    """
    Server-Sent Events stream.
    First event: {"type":"sources","sources":[...]}
    Then:       {"type":"token","content":"..."}
    Final:      {"type":"done"}
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    try:
        sources, tokens = rag_service.stream_answer(
            question, user.id, request.history
        )
    except Exception as exc:
        logger.exception("Stream query failed for user %s", user.id)
        raise HTTPException(500, f"Query failed: {exc}") from exc

    def event_gen():
        payload = {
            "type": "sources",
            "sources": [s.model_dump() for s in sources],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        try:
            for token in tokens:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
