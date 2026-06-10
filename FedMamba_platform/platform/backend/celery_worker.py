from celery import Celery
from celery.signals import worker_ready
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "fedmamba_salt",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.autodiscover_tasks(["app.tasks"])


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Run startup recovery when the Celery worker comes online."""
    from app.tasks.fl_tasks import recover_orphaned_experiments
    recover_orphaned_experiments.delay()
