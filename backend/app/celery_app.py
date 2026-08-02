"""
Celery application using Upstash Redis as broker + result backend.

Windows note: run the worker with the solo pool (no fork):
  celery -A app.celery_app.celery worker --pool=solo -l info

Use the Upstash *Redis* URL (rediss://...), not the REST URL.
"""

from __future__ import annotations

import logging
import ssl

from celery import Celery

from app.config import get_settings

logger = logging.getLogger(__name__)


def _ssl_cert_reqs(value: str):
    mapping = {
        "none": ssl.CERT_NONE,
        "optional": ssl.CERT_OPTIONAL,
        "required": ssl.CERT_REQUIRED,
    }
    return mapping.get((value or "none").lower(), ssl.CERT_NONE)


def _build_celery() -> Celery:
    settings = get_settings()
    broker = settings.redis_url
    backend = settings.redis_url

    app = Celery("genai_rag", broker=broker, backend=backend)

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        result_expires=3600,
        task_time_limit=settings.celery_task_time_limit,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        # Keep Upstash connection/command usage modest
        broker_pool_limit=2,
        broker_connection_retry_on_startup=True,
        broker_transport_options={"visibility_timeout": settings.celery_task_time_limit + 60},
        result_backend_transport_options={"visibility_timeout": settings.celery_task_time_limit + 60},
    )

    # TLS for Upstash (rediss://)
    if broker and broker.startswith("rediss://"):
        ssl_opts = {"ssl_cert_reqs": _ssl_cert_reqs(settings.redis_ssl_cert_reqs)}
        app.conf.broker_use_ssl = ssl_opts
        app.conf.redis_backend_use_ssl = ssl_opts
        logger.info(
            "Celery Redis TLS enabled (ssl_cert_reqs=%s)",
            settings.redis_ssl_cert_reqs,
        )

    # Ensure tasks are registered
    app.autodiscover_tasks(["app.tasks"])
    return app


celery = _build_celery()

# Import task modules so they register on the app
from app.tasks import ingest_tasks  # noqa: E402,F401
