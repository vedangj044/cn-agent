import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
import os
from utils.base import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///k8s_analysis.db")

engine = create_engine(DATABASE_URL, echo=True, future=True)
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

# Initialize database
def init_db():
    """Initialize SQLAlchemy database"""
    import models

    Base.metadata.create_all(engine)
    logging.info("Database initialized successfully")
