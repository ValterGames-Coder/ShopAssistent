import json

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .knowledge import fingerprint
from .llm import validate_vectors
from .models import Document, Hit, Knowledge


class IndexNotReady(RuntimeError):
    pass


SCHEMA = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS kb_documents (
        id text PRIMARY KEY,
        document jsonb NOT NULL,
        embedding vector NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS kb_metadata (
        singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
        fingerprint text NOT NULL,
        dimension integer NOT NULL CHECK (dimension > 0),
        updated_at timestamptz NOT NULL DEFAULT now()
    )""",
)


class PgVectorStore:
    def __init__(self, database_url: str, expected_fingerprint: str):
        self.database_url = database_url
        self.expected_fingerprint = expected_fingerprint

    async def connect(self):
        return await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row, connect_timeout=10,
            options="-c statement_timeout=15000",
        )

    async def replace(self, knowledge: Knowledge, model: str, vectors: list[list[float]]):
        validate_vectors(vectors, len(knowledge.documents))
        async with await self.connect() as conn:
            # One transaction: readers see the complete old or new revision.
            await conn.execute("SELECT pg_advisory_xact_lock(28184601)")
            for statement in SCHEMA:
                await conn.execute(statement)
            await conn.execute("DELETE FROM kb_documents")
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    "INSERT INTO kb_documents(id, document, embedding) VALUES (%s, %s, %s::vector)",
                    [(d.id, Jsonb(d.model_dump()), json.dumps(v))
                     for d, v in zip(knowledge.documents, vectors, strict=True)],
                )
            await conn.execute("""INSERT INTO kb_metadata(singleton, fingerprint, dimension)
                VALUES (true, %s, %s) ON CONFLICT (singleton) DO UPDATE
                SET fingerprint=excluded.fingerprint, dimension=excluded.dimension, updated_at=now()""",
                (fingerprint(knowledge, model), len(vectors[0])),
            )

    async def search(self, vector: list[float], top_k: int, min_score: float) -> list[Hit]:
        validate_vectors([vector], 1)
        async with await self.connect() as conn:
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cursor = await conn.execute("SELECT fingerprint, dimension FROM kb_metadata WHERE singleton=true")
            meta = await cursor.fetchone()
            if not meta or meta["fingerprint"] != self.expected_fingerprint or meta["dimension"] != len(vector):
                raise IndexNotReady("Rebuild the knowledge index with the configured embedding model")
            cursor = await conn.execute("""SELECT document, 1 - (embedding <=> %s::vector) AS score
                FROM kb_documents ORDER BY embedding <=> %s::vector LIMIT %s""",
                (json.dumps(vector), json.dumps(vector), top_k))
            rows = await cursor.fetchall()
        # Exact cosine search is intentional: just six documents, no approximate index needed.
        return [Hit(Document.model_validate(row["document"]), float(row["score"]))
                for row in rows if row["score"] >= min_score]
