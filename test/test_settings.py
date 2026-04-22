import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from server.settings import DEFAULT_PROJECT_ROOT, Settings, load_settings


class SettingsTests(unittest.TestCase):
    def test_settings_resolve_relative_paths(self):
        with patch.dict(
            os.environ,
            {
                "CUSTOM_MODEL_PATH": "./tts_models",
                "SPEAKER_PROFILES_PATH": "./speaker_profiles",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(
            settings.custom_model_path, (DEFAULT_PROJECT_ROOT / "tts_models").resolve()
        )
        self.assertEqual(
            settings.speaker_profiles_path,
            (DEFAULT_PROJECT_ROOT / "speaker_profiles").resolve(),
        )

    def test_settings_validate_scheduler_bounds(self):
        with patch.dict(os.environ, {"MAX_BATCH_SIZE": "0"}, clear=False):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_load_settings_clamps_gpu_worker_count(self):
        with patch.dict(os.environ, {"GPU_WORKER_COUNT": "3"}, clear=False):
            settings = load_settings()

        self.assertEqual(settings.gpu_worker_count, 1)

    def test_settings_ignore_unrelated_environment_variables(self):
        with patch.dict(
            os.environ,
            {
                "COQUI_TOS_AGREED": "1",
                "XTTS_PORT": "8003",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.num_threads, os.cpu_count() or 1)


if __name__ == "__main__":
    unittest.main()
