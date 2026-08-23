from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.base import Base

logger = get_logger("app.db.session")


class DatabaseManager:
    """Manages asynchronous SQLAlchemy database engine and connection lifecycles."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def is_configured(self) -> bool:
        """Check whether a database URL is configured."""
        return bool(self._settings.database_url and "postgres" in self._settings.database_url)

    @property
    def engine(self) -> AsyncEngine:
        """Return the active AsyncEngine instance, initializing if needed."""
        if self._engine is None:
            db_url = self._settings.database_url
            safe_url = db_url.split("@")[-1] if "@" in db_url else db_url
            logger.info("initializing_database_engine", url=safe_url)
            self._engine = create_async_engine(
                db_url,
                echo=self._settings.database_echo,
                pool_size=self._settings.database_pool_size,
                max_overflow=self._settings.database_max_overflow,
            )
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Return the active session factory."""
        if self._session_factory is None:
            _ = self.engine
        assert self._session_factory is not None
        return self._session_factory

    async def init_db(self) -> None:
        """Initialize database schema, enabling pgvector extension and creating tables."""
        if not self.is_configured:
            logger.warning("database_not_configured_skipping_init")
            return

        try:
            async with self.engine.begin() as conn:
                logger.info("enabling_pgvector_extension")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("creating_database_tables")
                await conn.run_sync(Base.metadata.create_all)
            logger.info("database_init_complete")
        except Exception as exc:
            logger.warning("database_init_skipped_or_failed", error=str(exc))

    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Provide an asynchronous transactional database session context."""
        session: AsyncSession = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def close(self) -> None:
        """Dispose of the database connection pool."""
        if self._engine is not None:
            logger.info("closing_database_engine")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
