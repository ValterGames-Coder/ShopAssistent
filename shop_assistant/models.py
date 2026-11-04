from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    title: str
    text: str
    max_days: int | None = Field(default=None, ge=1)


class Knowledge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int
    documents: list[Document]


class Decision(BaseModel):
    """No model-generated prose is sent to Telegram."""
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["answer", "unknown", "order_unknown"]
    source_ids: list[str] = Field(max_length=6)
    missing_information: bool
    return_condition: Literal["none", "unspecified", "opened_nondefective", "defective"]
    days_quote: str | None


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    source_ids: tuple[str, ...] = ()
    status: str = "answered"
