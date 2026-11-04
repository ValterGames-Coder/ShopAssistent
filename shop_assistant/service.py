import logging
import time

import httpx
import psycopg
from pydantic import ValidationError

from .config import Settings
from .llm import ModelError
from .models import Answer, Hit, Knowledge
from .policy import ORDER_UNKNOWN, UNAVAILABLE, UNKNOWN, has_defect, is_specific_order, mentions_discount, mentions_return, render
from .store import IndexNotReady

logger = logging.getLogger(__name__)


class Assistant:
    def __init__(self, settings: Settings, knowledge: Knowledge, model, store, prompt: str):
        self.settings, self.model, self.store, self.prompt = settings, model, store, prompt
        self.catalog = {doc.id: doc for doc in knowledge.documents}

    async def answer(self, question: str) -> Answer:
        started = time.monotonic()
        if not isinstance(question, str) or not question.strip() or len(question) > self.settings.max_question_length:
            return Answer(f"Напишите вопрос длиной от 1 до {self.settings.max_question_length} символов.", status="invalid")
        # No order database exists. Even a compromised LLM cannot provide order details.
        if is_specific_order(question):
            return Answer(ORDER_UNKNOWN, status="order_unknown")
        try:
            vectors = await self.model.embed([question])
            hits = await self.store.search(vectors[0], self.settings.retrieval_top_k,
                                           self.settings.retrieval_min_score)
            if not hits:
                return Answer(UNKNOWN, status="unknown")
            # Expand linked policy clauses so retrieval cannot separate a rule from its exception.
            ids = {hit.document.id for hit in hits}
            related = set()
            if ids & {"return", "opened_return", "defect"} or mentions_return(question) or has_defect(question):
                related |= {"return", "opened_return", "defect"}
            if mentions_discount(question):
                related.add("discount")
            hits += [Hit(self.catalog[key], 1.0) for key in sorted(related - ids)]
            decision = await self.model.decide(question, hits, self.prompt)
            result = render(question, decision, hits, self.catalog)
            logger.info("answer status=%s sources=%s elapsed_ms=%d", result.status,
                        ",".join(result.source_ids), int((time.monotonic() - started) * 1000))
            return result
        except (ValidationError, ModelError, ValueError, KeyError, TypeError) as exc:
            logger.warning("invalid_model_response error_type=%s", type(exc).__name__)
            return Answer(UNKNOWN, status="rejected")
        except (httpx.HTTPError, psycopg.Error, IndexNotReady) as exc:
            # Never log exception bodies: URLs and request content may contain secrets.
            logger.warning("dependency_unavailable error_type=%s", type(exc).__name__)
            return Answer(UNAVAILABLE, status="unavailable")
