import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import QueryRequest, QueryResponse
from app.services import rag_service
from app.services.auth_service import User, get_current_user
from app.services.conversation_service import (
    get_or_create_conversation,
    load_history,
    save_turn,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


def _resolve_history(
    user_id: str,
    conversation_id: str | None,
    inline_history: list,
) -> tuple[str, list]:
    """Load history from Postgres unless the client sent inline turns."""
    conv_id = get_or_create_conversation(user_id, conversation_id)
    if inline_history:
        return conv_id, inline_history
    return conv_id, load_history(conv_id, user_id)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    user: User = Depends(get_current_user),
):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    try:
        conv_id, history = _resolve_history(
            user.id, request.conversation_id, request.history
        )
        answer, sources, meta = rag_service.answer_question(
            question, user.id, history
        )
        save_turn(
            conv_id,
            user.id,
            question,
            answer,
            citations=[s.model_dump() for s in sources],
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            conversation_id=conv_id,
            retrieval_query=meta.get("retrieval_query"),
            query_rewritten=meta.get("query_rewritten", False),
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
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
    First event: {"type":"sources","sources":[...],"conversation_id":"..."}
    Then:       {"type":"token","content":"..."}
    Final:      {"type":"done","conversation_id":"...","retrieval_query":"..."}
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    try:
        conv_id, history = _resolve_history(
            user.id, request.conversation_id, request.history
        )
        sources, tokens, meta = rag_service.stream_answer(
            question, user.id, history
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Stream query failed for user %s", user.id)
        raise HTTPException(500, f"Query failed: {exc}") from exc

    collected: list[str] = []

    def event_gen():
        payload = {
            "type": "sources",
            "sources": [s.model_dump() for s in sources],
            "conversation_id": conv_id,
            "retrieval_query": meta.get("retrieval_query"),
            "query_rewritten": meta.get("query_rewritten", False),
        }
        yield f"data: {json.dumps(payload)}\n\n"
        try:
            for token in tokens:
                collected.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        else:
            full_answer = "".join(collected)
            try:
                save_turn(
                    conv_id,
                    user.id,
                    question,
                    full_answer,
                    citations=[s.model_dump() for s in sources],
                )
            except Exception:
                logger.exception("Failed to persist stream turn for %s", conv_id)
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'retrieval_query': meta.get('retrieval_query'), 'query_rewritten': meta.get('query_rewritten', False)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
