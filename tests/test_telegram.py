import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import GetMe, SendMessage
from aiogram.types import Chat, Message, Update, User

from shop_assistant.config import ROOT, Settings
from shop_assistant.knowledge import load_knowledge
from shop_assistant.models import Answer
from shop_assistant.telegram import Cooldown, build_router


class FakeTelegramSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.sent = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        if isinstance(method, GetMe):
            return User(id=123456, is_bot=True, first_name="Test", username="test_shop_bot")
        if isinstance(method, SendMessage):
            self.sent.append(method)
            return Message(message_id=1, date=datetime.now(timezone.utc),
                           chat=Chat(id=method.chat_id, type="private"), text=method.text)
        return True

    async def stream_content(self, url, **kwargs):
        if False:
            yield b""


class TelegramTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assistant = AsyncMock()
        self.assistant.answer.return_value = Answer("Наушники X1 в наличии. Цена — 2490 ₽.")
        self.session = FakeTelegramSession()
        self.bot = Bot("123456:fake_token_for_offline_tests", session=self.session)
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(build_router(self.assistant, Settings(database_url="unused"),
                                                     load_knowledge(ROOT / "knowledge.json")))

    async def asyncTearDown(self):
        await self.bot.session.close()

    async def feed(self, text=None, chat_type="private"):
        values = {"message_id": 1, "date": datetime.now(timezone.utc),
                  "chat": Chat(id=42, type=chat_type),
                  "from_user": User(id=42, is_bot=False, first_name="Test")}
        if text is not None:
            values["text"] = text
        await self.dispatcher.feed_update(self.bot, Update(update_id=1, message=Message(**values)))

    async def test_text_goes_to_assistant_and_is_plain_text(self):
        await self.feed("Почём X1?")
        self.assistant.answer.assert_awaited_once_with("Почём X1?")
        self.assertEqual(self.session.sent[-1].text, self.assistant.answer.return_value.text)
        self.assertIsNone(self.session.sent[-1].parse_mode)

    async def test_start_and_sources_work_without_llm(self):
        await self.feed("/start")
        self.assertIn("Каждый вопрос", self.session.sent[-1].text)
        await self.feed("/sources")
        self.assertIn("Скидок нет", self.session.sent[-1].text)
        self.assistant.answer.assert_not_awaited()

    async def test_unknown_command(self):
        await self.feed("/admin")
        self.assertIn("Доступные команды", self.session.sent[-1].text)
        self.assistant.answer.assert_not_awaited()

    async def test_unsupported_media(self):
        await self.feed()
        self.assertIn("только текст", self.session.sent[-1].text)

    async def test_groups_are_ignored(self):
        await self.feed("X1?", chat_type="group")
        self.assertEqual(self.session.sent, [])
        self.assistant.answer.assert_not_awaited()

    async def test_throttling(self):
        await self.feed("Цена?")
        await self.feed("Цена?")
        self.assertEqual(self.assistant.answer.await_count, 1)
        self.assertIn("подождите", self.session.sent[-1].text)

    def test_cooldown_expires_and_storage_is_bounded(self):
        cooldown = Cooldown(3, capacity=2)
        self.assertTrue(cooldown.allow(1, now=0))
        self.assertFalse(cooldown.allow(1, now=2))
        self.assertTrue(cooldown.allow(1, now=3))
        self.assertTrue(cooldown.allow(2, now=3))
        self.assertTrue(cooldown.allow(3, now=3))
        self.assertEqual(len(cooldown.last_seen), 2)
