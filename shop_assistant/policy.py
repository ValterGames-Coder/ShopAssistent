"""Validate model decisions and render only trusted policy text."""
import re
import unicodedata

from .models import Answer, Decision, Document, Hit

UNKNOWN = "В базе знаний нет сведений, чтобы ответить на этот вопрос."
ORDER_UNKNOWN = "В базе знаний нет данных о конкретных заказах. Статус и время доставки этого заказа неизвестны."
UNAVAILABLE = "Сейчас не удалось проверить базу знаний. Попробуйте задать вопрос позже."


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def is_specific_order(text: str) -> bool:
    text = normalize(text)
    return bool(re.search(
        r"(?:заказ\w*|order)\s*(?:№|#|no\.?|номер)?\s*\d+|"
        r"(?:мо(?:й|я|е|и|его|ему|ем|их|ю|ей)|наш\w*)\s+заказ|(?:доставка|статус|состояние|оплата)\s+заказа|"
        r"(?:мой|моя|моего)\s+(?:платеж|оплата|возврат)|my\s+order", text))


def has_defect(text: str) -> bool:
    text = normalize(text)
    text = re.sub(r"без\s+брака|не\s+бракован\w*|нет\s+брака|без\s+дефект\w*", "", text)
    return bool(re.search(r"брак|сломан|не\s+работа|дефект|неисправ|broken|defective", text))


def mentions_return(text: str) -> bool:
    return bool(re.search(r"верну|возврат|return|refund", normalize(text)))


def mentions_discount(text: str) -> bool:
    return bool(re.search(r"скид|промокод|discount|coupon", normalize(text)))


def mentions_opened(text: str) -> bool:
    return bool(re.search(r"вскры|распак|откры|opened|unpacked", normalize(text)))


def parse_days(text: str) -> set[int]:
    text = normalize(text)
    days = {int(m) for m in re.findall(r"(?<!\d)(\d{1,5})(?:\s*[-‐‑–]?\s*(?:й|ый|ой|го))?\s*(?:день|дня|дней|суток|сутки|days?)\b", text)}
    # Shared unit: «на 14-й или 16-й день» names two different durations.
    days.update(int(m) for m in re.findall(
        r"(?<!\d)(\d{1,5})(?:\s*[-‐‑–]?\s*(?:й|ый|ой|го))?\s*(?:или|и|/)\s*\d{1,5}(?:\s*[-‐‑–]?\s*(?:й|ый|ой|го))?\s*(?:день|дня|дней|days?)\b", text))
    words = {"четырнадцат": 14, "пятнадцат": 15, "шестнадцат": 16,
             "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19,
             "двадцат": 20, "три недели": 21, "две недели": 14}
    for word, number in words.items():
        pattern = re.escape(word) if " " in word else rf"\b{word}\w*\s+(?:день|дня|дней)\b"
        if re.search(pattern, text):
            days.add(number)
    return days


def render(question: str, decision: Decision, hits: list[Hit], catalog: dict[str, Document]) -> Answer:
    available = {hit.document.id: hit.document for hit in hits}
    # Validate the whole selection, never silently discard fabricated citations.
    if any(source not in available for source in decision.source_ids):
        return Answer(UNKNOWN, status="rejected")
    if decision.days_quote is not None and decision.days_quote not in question:
        return Answer(UNKNOWN, status="rejected")
    if is_specific_order(question) or decision.kind == "order_unknown":
        return Answer(ORDER_UNKNOWN, status="order_unknown")

    ids = list(dict.fromkeys(decision.source_ids)) if decision.kind == "answer" else []
    # These narrow checks enforce constraints; semantic routing belongs to the LLM.
    if mentions_discount(question):
        if "discount" not in available:
            return Answer(UNKNOWN, status="unknown")
        if "discount" not in ids:
            ids.append("discount")

    defective = has_defect(question) or decision.return_condition == "defective"
    returning = mentions_return(question) or decision.return_condition != "none"
    if defective:
        ids = [source for source in ids if source not in {"return", "opened_return"}]
        if "defect" not in available:
            return Answer(UNKNOWN, status="unknown")
        if "defect" not in ids:
            ids.append("defect")
    elif returning and mentions_opened(question):
        if "opened_return" not in available:
            return Answer(UNKNOWN, status="unknown")
        ids = [source for source in ids if source != "return"]
        if "opened_return" not in ids:
            ids.append("opened_return")

    if not ids:
        return Answer(UNKNOWN, status="unknown")
    paragraphs = []
    days = parse_days(question) if returning else set()
    for source in ids:
        doc = available[source]
        # Compare against trusted on-disk content as well as the indexed revision.
        if source not in catalog or doc != catalog[source]:
            return Answer(UNKNOWN, status="rejected")
        if source == "delivery":
            paragraphs.append("Общие условия доставки: " + doc.text)
        elif source == "return":
            paragraphs.append("Общие условия возврата: " + doc.text)
        else:
            paragraphs.append(doc.text)
        if source in {"return", "opened_return"} and days:
            if len(days) != 1:
                paragraphs.append("Уточните, сколько дней прошло: в вопросе указано несколько сроков.")
            elif doc.max_days is not None and next(iter(days)) > doc.max_days:
                day = next(iter(days))
                paragraphs.append(f"На {day}-й день срок возврата товара без брака по этому правилу уже превышен.")
    if decision.missing_information:
        paragraphs.append("Для остальной части вопроса в базе знаний нет сведений.")
    return Answer("\n\n".join(paragraphs), tuple(ids))
