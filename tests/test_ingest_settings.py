"""Tests for ingest settings loaded from environment variables."""
import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.core.config import Settings


class IngestSettingsTests(unittest.TestCase):
    def test_embedding_batch_size_is_loaded_from_environment(self):
        with patch.dict(os.environ, {"INGEST_EMBEDDING_BATCH_SIZE": "7"}):
            settings = Settings(_env_file=None)
        self.assertEqual(settings.INGEST_EMBEDDING_BATCH_SIZE, 7)

    def test_embedding_batch_size_must_be_positive(self):
        with patch.dict(os.environ, {"INGEST_EMBEDDING_BATCH_SIZE": "0"}):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)


if __name__ == "__main__":
    unittest.main()
