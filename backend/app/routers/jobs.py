"""Job status endpoint for async ingest tasks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.schemas import JobStatus, UploadItem
from app.services.auth_service import User, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, user: User = Depends(get_current_user)):
    settings = get_settings()
    if not settings.async_ingest or not settings.redis_url:
        raise HTTPException(503, "Async ingest is disabled.")

    from celery.result import AsyncResult

    from app.celery_app import celery

    res = AsyncResult(job_id, app=celery)
    state = res.state
    info = res.info

    status = JobStatus(job_id=job_id, state=state)

    if state == "PROGRESS" and isinstance(info, dict):
        status.stage = info.get("stage")
        status.filename = info.get("filename")
    elif state == "SUCCESS" and isinstance(info, dict):
        status.stage = "done"
        status.filename = info.get("filename")
        try:
            status.result = UploadItem(**info)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Job %s result did not match UploadItem: %s", job_id, info)
    elif state == "FAILURE":
        status.stage = "failed"
        status.error = str(info) if info else "Task failed"

    return status
