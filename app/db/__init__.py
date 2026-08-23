"""Database and Vector persistence package utilizing PostgreSQL with pgvector."""

from app.db.base import Base
from app.db.models import DocumentChunkRecord, DocumentRecord
from app.db.repository import (
    BaseVectorRepository,
    InMemoryVectorRepository,
    PgVectorRepository,
    create_vector_repository,
)
from app.db.session import DatabaseManager

__all__ = [
    "Base",
    "BaseVectorRepository",
    "DatabaseManager",
    "DocumentChunkRecord",
    "DocumentRecord",
    "InMemoryVectorRepository",
    "PgVectorRepository",
    "create_vector_repository",
]
