import argparse
import asyncio
import json
import logging
from pathlib import Path

from .config import ROOT, Settings
from .evaluation import evaluate
from .knowledge import fingerprint, load_knowledge
from .llm import Ollama
from .service import Assistant
from .store import PgVectorStore
from .telegram import run_bot


async def run(args):
    settings = Settings.from_env()
    knowledge = load_knowledge(ROOT / "knowledge.json")
    revision = fingerprint(knowledge, settings.embedding_model)
    store = PgVectorStore(settings.database_url, revision)
    model = Ollama(settings)
    assistant = Assistant(settings, knowledge, model, store,
                          (ROOT / "SYSTEM_PROMPT.md").read_text(encoding="utf-8"))
    try:
        if args.command == "index":
            vectors = await model.embed([doc.title + "\n" + doc.text for doc in knowledge.documents])
            await store.replace(knowledge, settings.embedding_model, vectors)
            print(f"Indexed {len(vectors)} documents; dimensions={len(vectors[0])}")
        elif args.command == "bot":
            await run_bot(assistant, settings, knowledge)
        elif args.command == "ask":
            result = await assistant.answer(args.question)
            print(json.dumps({"answer": result.text, "source_ids": result.source_ids, "status": result.status}, ensure_ascii=False, indent=2))
            if result.status in {"rejected", "unavailable", "invalid"}:
                return 1
        elif args.command == "evaluate":
            report = await evaluate(assistant, settings)
            payload = json.dumps(report, ensure_ascii=False, indent=2)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload + "\n", encoding="utf-8")
            print(payload)
            return 0 if report["all_passed"] else 1
        return 0
    finally:
        await model.close()


def main():
    parser = argparse.ArgumentParser(description="Telegram shop assistant: local LLM + vector RAG")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Create or rebuild the knowledge index")
    commands.add_parser("bot", help="Run Telegram long polling")
    ask = commands.add_parser("ask", help="Ask a single question using the real RAG pipeline")
    ask.add_argument("question")
    evaluation = commands.add_parser("evaluate", help="Run the three live crash tests")
    evaluation.add_argument("--output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        code = asyncio.run(run(args))
    except KeyboardInterrupt:
        code = 0
    except Exception as exc:
        # Do not expose tokens, DSNs, user text, or provider response bodies in logs.
        logging.error("startup_failed error_type=%s; check .env, Ollama, index and model downloads", type(exc).__name__)
        code = 1
    raise SystemExit(code)
