import hashlib
import json
from pathlib import Path

from .models import Knowledge

REQUIRED_IDS = {"product", "delivery", "return", "opened_return", "defect", "discount"}


def load_knowledge(path: Path) -> Knowledge:
    knowledge = Knowledge.model_validate_json(path.read_text(encoding="utf-8"))
    ids = [d.id for d in knowledge.documents]
    if len(ids) != len(set(ids)) or set(ids) != REQUIRED_IDS:
        raise ValueError("Knowledge must contain the six distinct shop policy documents")
    for document in knowledge.documents:
        if not document.text.strip() or not document.title.strip():
            raise ValueError("Empty knowledge document")
        if document.id in {"return", "opened_return"}:
            if document.max_days is None or str(document.max_days) not in document.text:
                raise ValueError("Return limit must agree with the policy text")
    return knowledge


def fingerprint(knowledge: Knowledge, embedding_model: str) -> str:
    payload = {"knowledge": knowledge.model_dump(), "embedding_model": embedding_model,
               "embedding_format": "title-newline-text-v1"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
