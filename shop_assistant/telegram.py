import asyncio
import logging
import time
from collections import OrderedDict

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import ErrorEvent, Message
from aiogram.utils.chat_action import ChatActionSender

from .config import Settings
from .models import Knowledge
from .policy import UNAVAILABLE

logger = logging.getLogger(__name__)


class Cooldown:
    def __init__(self, seconds: float, capacity: int = 4096):
        self.seconds, self.capacity = seconds, capacity
        self.last_seen: OrderedDict[int, float] = OrderedDict()

    def allow(self, user_id: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        previous = self.last_seen.get(user_id)
        if previous is not None and now - previous < self.seconds:
            return False
        self.last_seen[user_id] = now
        self.last_seen.move_to_end(user_id)
        while len(self.last_seen) > self.capacity:
            self.last_seen.popitem(last=False)
        return True


def build_router(assistant, settings: Settings, knowledge: Knowledge) -> Router:
    router = Router()
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    cooldown = Cooldown(settings.user_cooldown_seconds)
    # A small CPU model handles one request at a time; no unbounded request queue.
    lock = asyncio.Lock()

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def start(message: Message):
        await message.answer(
            "Здравствуйте! Я ассистент магазина наушников X1. Отвечаю по регламенту "
            "о товаре, доставке, возврате и скидках. Данных о конкретных заказах у меня нет.\n\n"
            "Каждый вопрос рассматриваю отдельно — укажите все условия в одном сообщении.\n"
            "Например: «Можно вернуть распакованные наушники через 16 дней?»\n"
            "/sources — показать базу знаний."
        )

    @router.message(Command("sources"))
    async def sources(message: Message):
        await message.answer("Регламент магазина:\n\n" + "\n\n".join(d.text for d in knowledge.documents))

    @router.message(F.text)
    async def question(message: Message):
        if not message.from_user:
            return
        if message.text.startswith("/"):
            await message.answer("Доступные команды: /start, /help, /sources. Или напишите вопрос о магазине.")
            return
        if not cooldown.allow(message.from_user.id):
            await message.answer("Пожалуйста, подождите несколько секунд перед следующим вопросом.")
            return
        if lock.locked():
            await message.answer("Сейчас обрабатываю другой вопрос. Попробуйте немного позже.")
            return
        async with lock:
            async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
                result = await assistant.answer(message.text)
                # Plain text: generated or user-controlled HTML is never interpreted.
                await message.answer(result.text, parse_mode=None)

    @router.message()
    async def unsupported(message: Message):
        await message.answer("Пока я понимаю только текст. Напишите вопрос сообщением.")

    @router.error()
    async def error(event: ErrorEvent):
        logger.error("telegram_handler_error error_type=%s", type(event.exception).__name__)
        if event.update.message:
            try:
                await event.update.message.answer(UNAVAILABLE)
            except Exception as exc:
                logger.warning("telegram_reply_failed error_type=%s", type(exc).__name__)
        return True

    return router


async def run_bot(assistant, settings: Settings, knowledge: Knowledge):
    if not settings.telegram_bot_token:
        raise ValueError("Set TELEGRAM_BOT_TOKEN in .env before starting the bot")
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(assistant, settings, knowledge))
    session = AiohttpSession(proxy=settings.telegram_proxy_url or None)
    async with Bot(settings.telegram_bot_token, session=session) as bot:
        # Do not delete an existing webhook or discard pending updates silently.
        webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ValueError("This bot has an active webhook. Use a dedicated bot for local polling.")
        logger.info("telegram_polling_started")
        await dispatcher.start_polling(bot, allowed_updates=["message"], tasks_concurrency_limit=8)
