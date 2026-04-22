import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server.schemas import StreamingInputs, TTSInputs
from server.routes.tts import _resolve_speaker_data
from server.speaker_profiles import SpeakerProfileStore


class SpeakerProfileStoreTests(unittest.TestCase):
    def test_save_list_get_and_delete_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(Path(temp_dir))
            profile = store.save_profile(
                "demo_voice",
                speaker_embedding=[0.1, 0.2, 0.3],
                gpt_cond_latent=[[1.0] * 1024],
                name="Demo Voice",
                description="Test profile",
            )

            self.assertEqual(profile["id"], "demo_voice")
            self.assertEqual(len(store.list_profiles()), 1)
            loaded = store.get_profile("demo_voice")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["name"], "Demo Voice")
            self.assertTrue(store.delete_profile("demo_voice"))
            self.assertEqual(store.list_profiles(), [])

    def test_save_profile_rejects_invalid_profile_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(Path(temp_dir))
            with self.assertRaises(ValueError):
                store.save_profile(
                    "bad id",
                    speaker_embedding=[0.1],
                    gpt_cond_latent=[[1.0] * 1024],
                )

    def test_list_profiles_skips_invalid_profile_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(Path(temp_dir))
            broken_profile = Path(temp_dir) / "broken.json"
            broken_profile.write_text("{not json}", encoding="utf8")

            self.assertEqual(store.list_profiles(), [])

    def test_legacy_folder_profile_is_listed_loaded_and_deleted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(Path(temp_dir))
            legacy_dir = Path(temp_dir) / "fahd"
            legacy_dir.mkdir()
            (legacy_dir / "speaker_embedding.json").write_text("[0.1, 0.2, 0.3]", encoding="utf8")
            (legacy_dir / "gpt_cond_latent.json").write_text(
                "[[" + ", ".join(["1.0"] * 1024) + "]]",
                encoding="utf8",
            )

            profiles = store.list_profiles()
            loaded = store.get_profile("fahd")

            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0]["id"], "fahd")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["name"], "fahd")
            self.assertTrue(store.has_profile("fahd"))
            self.assertTrue(store.delete_profile("fahd"))
            self.assertFalse(legacy_dir.exists())

    def test_nested_generated_profile_directory_is_not_treated_as_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SpeakerProfileStore(Path(temp_dir))
            generated_dir = Path(temp_dir) / "xtts-arabic-sa"
            generated_dir.mkdir()
            (generated_dir / "001_030.json").write_text("{}", encoding="utf8")

            self.assertEqual(store.list_profiles(), [])
            self.assertIsNone(store.get_profile("xtts-arabic-sa"))


class SpeakerProfileSchemaTests(unittest.TestCase):
    def test_tts_inputs_accept_speaker_profile_id_without_latents(self):
        payload = TTSInputs(
            text="hello",
            language="en",
            speaker_profile_id="demo_voice",
        )
        self.assertEqual(payload.speaker_profile_id, "demo_voice")

    def test_streaming_inputs_still_accept_raw_latents(self):
        payload = StreamingInputs(
            text="hello",
            language="en",
            speaker_embedding=[0.1, 0.2],
            gpt_cond_latent=[[1.0] * 1024],
        )
        self.assertIsNone(payload.speaker_profile_id)
        self.assertEqual(payload.stream_chunk_size, 20)

    def test_tts_inputs_allow_server_default_flow_without_voice_fields(self):
        payload = TTSInputs(
            text="hello",
            language="en",
        )
        self.assertIsNone(payload.speaker_profile_id)
        self.assertIsNone(payload.speaker_embedding)
        self.assertIsNone(payload.gpt_cond_latent)

    def test_tts_inputs_require_embedding_and_latent_together(self):
        with self.assertRaises(ValueError):
            TTSInputs(
                text="hello",
                language="en",
                speaker_embedding=[0.1, 0.2],
            )


class SpeakerResolutionTests(unittest.TestCase):
    def test_direct_embeddings_take_precedence_over_profile_id(self):
        parsed_input = TTSInputs(
            text="hello",
            language="en",
            speaker_profile_id="demo_voice",
            speaker_embedding=[0.1, 0.2],
            gpt_cond_latent=[[1.0] * 1024],
        )
        fake_state = SimpleNamespace(
            settings=SimpleNamespace(default_speaker_profile_id=None),
            speaker_profiles=SimpleNamespace(has_profile=lambda profile_id: False),
            service=SimpleNamespace(get_default_speaker=lambda: None),
        )

        with patch("server.routes.tts.get_app_state", return_value=fake_state):
            speaker_embedding, gpt_cond_latent = _resolve_speaker_data(parsed_input, request=None)

        self.assertEqual(speaker_embedding, [0.1, 0.2])
        self.assertEqual(gpt_cond_latent, [[1.0] * 1024])

    def test_server_default_profile_is_used_when_request_has_no_voice(self):
        parsed_input = TTSInputs(
            text="hello",
            language="en",
        )
        fake_profile_store = SimpleNamespace(
            get_profile=lambda profile_id: {
                "speaker_embedding": [0.5, 0.6],
                "gpt_cond_latent": [[2.0] * 1024],
            },
            has_profile=lambda profile_id: False,
        )
        fake_state = SimpleNamespace(
            settings=SimpleNamespace(default_speaker_profile_id="server_default"),
            speaker_profiles=fake_profile_store,
            service=SimpleNamespace(get_default_speaker=lambda: None),
        )

        with patch("server.routes.tts.get_app_state", return_value=fake_state):
            speaker_embedding, gpt_cond_latent = _resolve_speaker_data(parsed_input, request=None)

        self.assertEqual(speaker_embedding, [0.5, 0.6])
        self.assertEqual(gpt_cond_latent, [[2.0] * 1024])


if __name__ == "__main__":
    unittest.main()
