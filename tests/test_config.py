import os
import unittest
from unittest.mock import patch

from psycopg.conninfo import conninfo_to_dict

from shop_assistant.config import Settings


class DockerConfigTests(unittest.TestCase):
    @patch("shop_assistant.config.load_dotenv")
    def test_special_characters_in_database_password(self, _):
        with patch.dict(os.environ, {"PGPASSWORD": "quote' slash\\ space @:?"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(conninfo_to_dict(settings.database_url)["password"], "quote' slash\\ space @:?")
        self.assertEqual(settings.store_backend, "postgres")
        self.assertEqual(settings.ollama_base_url, "http://ollama:11434")

    def test_sqlite_is_not_silently_used(self):
        with self.assertRaises(ValueError):
            Settings(database_url="unused", store_backend="sqlite")
