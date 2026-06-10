from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.base import Base


settings = get_settings()

engine = create_engine(settings.DATABASE_URL, echo=False)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables in the database.

    Uses Base.metadata from the models package. All models must be imported
    before calling this (they are via app.models.__init__).
    """
    import app.models  # noqa: F401 — ensure all models are registered
    Base.metadata.create_all(bind=engine)
