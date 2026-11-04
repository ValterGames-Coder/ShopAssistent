"""Run against a dedicated test DB via TEST_DATABASE_URL, never the bot's DB."""
import os
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql

from shop_assistant.config import ROOT
from shop_assistant.knowledge import fingerprint, load_knowledge
from shop_assistant.store import IndexNotReady, PgVectorStore

DATABASE_URL = os.getenv("TEST_DATABASE_URL")


class IsolatedStore(PgVectorStore):
    def __init__(self, url, revision, schema):
        super().__init__(url, revision)
        self.schema = schema

    async def connect(self):
        conn = await super().connect()
        await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
        await conn.commit()
        return conn


@unittest.skipUnless(DATABASE_URL, "Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector database")
class PostgreSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema = "shop_test_" + uuid4().hex
        self.knowledge = load_knowledge(ROOT / "knowledge.json")
        self.revision = fingerprint(self.knowledge, "test-embedding")
        self.vectors = [[1., 0.], [.8, .6], [0., 1.], [-1., 0.], [0., -1.], [-.8, -.6]]
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.store = IsolatedStore(DATABASE_URL, self.revision, self.schema)
        await self.store.replace(self.knowledge, "test-embedding", self.vectors)

    async def asyncTearDown(self):
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            await conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))

    async def test_real_cosine_search(self):
        hits = await self.store.search([1., 0.], 2, .1)
        self.assertEqual([hit.document.id for hit in hits], ["product", "delivery"])
        self.assertAlmostEqual(hits[0].score, 1.)
        self.assertAlmostEqual(hits[1].score, .8, places=5)

    async def test_threshold(self):
        hits = await self.store.search([1., 0.], 6, .9)
        self.assertEqual([hit.document.id for hit in hits], ["product"])

    async def test_stale_revision_and_dimension(self):
        other = IsolatedStore(DATABASE_URL, "outdated", self.schema)
        with self.assertRaises(IndexNotReady):
            await other.search([1., 0.], 6, .3)
        with self.assertRaises(IndexNotReady):
            await self.store.search([1., 0., 0.], 6, .3)

    async def test_reindex_replaces_vectors(self):
        await self.store.replace(self.knowledge, "test-embedding", list(reversed(self.vectors)))
        hits = await self.store.search([1., 0.], 1, .9)
        self.assertEqual(hits[0].document.id, "discount")

    async def test_failed_import_rolls_back(self):
        duplicate_ids = self.knowledge.model_copy(update={"documents": [self.knowledge.documents[0]] * 6})
        with self.assertRaises(psycopg.errors.UniqueViolation):
            await self.store.replace(duplicate_ids, "test-embedding", self.vectors)
        hits = await self.store.search([1., 0.], 1, .9)
        self.assertEqual(hits[0].document.id, "product")
