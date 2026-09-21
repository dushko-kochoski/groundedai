from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import engine


@dataclass(frozen=True, slots=True)
class DatabaseReadiness:
    database: bool
    pgvector: bool

    @property
    def ready(self) -> bool:
        return self.database and self.pgvector


async def check_database_readiness() -> DatabaseReadiness:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            extension = await connection.scalar(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
    except (SQLAlchemyError, OSError):
        return DatabaseReadiness(database=False, pgvector=False)
    return DatabaseReadiness(database=True, pgvector=extension == 1)

