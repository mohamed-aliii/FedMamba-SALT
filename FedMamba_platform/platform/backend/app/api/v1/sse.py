"""
SSE (Server-Sent Events) endpoint for real-time experiment monitoring.

Subscribes to Redis pub/sub channel experiment:{id} and forwards
each published message as an SSE event to the connected browser.
"""
import asyncio
import json
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.deps import get_current_user
from app.database import get_db
from app.models.experiment import FLExperiment
from app.models.user import User

router = APIRouter(prefix="/sse", tags=["sse"])
settings = get_settings()


@router.get("/experiments/{experiment_id}")
async def sse_experiment(
    experiment_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Subscribe to real-time updates for an experiment.

    Delivers SSE events from Redis pub/sub channel experiment:{id}.
    Browser connects once; events are pushed as rounds complete.
    Client disconnects → subscription cleaned up automatically.
    """
    # Verify experiment exists
    experiment = db.get(FLExperiment, experiment_id)
    if experiment is None:
        async def _not_found():
            yield f"data: {json.dumps({'error': 'Experiment not found'})}\n\n"
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    channel = f"experiment:{experiment_id}"

    async def _event_stream():
        # Async Redis client for pub/sub (separate from sync client)
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        # Send initial connection confirmation
        yield f"data: {json.dumps({'type': 'connected', 'channel': channel})}\n\n"

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    data = message["data"]
                    yield f"data: {data}\n\n"

                    # If experiment is complete, close the stream
                    try:
                        parsed = json.loads(data)
                        if parsed.get("type") == "experiment_complete":
                            break
                    except json.JSONDecodeError:
                        pass

                await asyncio.sleep(0.1)

        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            await redis_client.aclose()

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )
