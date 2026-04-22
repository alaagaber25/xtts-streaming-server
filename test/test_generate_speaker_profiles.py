import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from generate_speaker_profiles import (
    collect_reference_files,
    derive_weights_name,
    generate_profiles,
    main,
    patch_xtts_audio_loader,
    resolve_model_dir,
)
from server.settings import DEFAULT_PROJECT_ROOT
from server.speaker_profiles import SpeakerProfileStore


class FakeXTTSModel:
    def get_conditioning_latents(self, audio_path: str):
        audio_name = Path(audio_path).name
        embedding_seed = float(len(audio_name))
        return [[embedding_seed] * 1024], [embedding_seed, embedding_seed + 1.0]


class FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def getChild(self, _name: str):
        return self


class GenerateSpeakerProfilesTests(unittest.TestCase):
    def test_torchaudio_version_mismatch_raises_actionable_error(self):
        from server.compat import ensure_torchaudio_compatibility

        with patch("server.compat.importlib_metadata.version", return_value="2.11.0"):
            with self.assertRaisesRegex(RuntimeError, "Installed torchaudio==2.11.0"):
                ensure_torchaudio_compatibility()

    def test_torchaudio_import_failure_is_wrapped_with_repair_hint(self):
        from server.compat import ensure_torchaudio_compatibility

        with patch("server.compat.importlib_metadata.version", return_value="2.5.1+cu121"), patch(
            "server.compat.importlib.import_module",
            side_effect=OSError(127, "The specified procedure could not be found"),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to import its native extension"):
                ensure_torchaudio_compatibility()

    def test_resolve_model_dir_accepts_direct_model_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "custom_weights"
            model_dir.mkdir()
            for filename in ("config.json", "model.pth", "vocab.json"):
                (model_dir / filename).write_text("{}", encoding="utf8")

            resolved = resolve_model_dir(model_dir / "model.pth")

            self.assertEqual(resolved, model_dir.resolve())

    def test_derive_weights_name_uses_parent_for_generic_model_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_arg = Path(temp_dir) / "xtts_server" / "model"
            weights_arg.mkdir(parents=True)

            derived = derive_weights_name(weights_arg.resolve(), weights_arg.resolve(), explicit_name=None)

            self.assertEqual(derived, "xtts_server")

    def test_collect_reference_files_filters_and_sorts_audio_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_dir = Path(temp_dir)
            for filename in ("b.wav", "a.mp3", "ignore.txt"):
                (reference_dir / filename).write_text("x", encoding="utf8")

            collected = collect_reference_files(reference_dir)

            self.assertEqual([path.name for path in collected], ["a.mp3", "b.wav"])

    def test_generate_profiles_writes_profiles_into_weight_specific_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "speaker_profiles" / "arabic_xtts"
            reference_dir = Path(temp_dir) / "references"
            reference_dir.mkdir()
            first_reference = reference_dir / "voice one.wav"
            second_reference = reference_dir / "voice@one.wav"
            first_reference.write_text("a", encoding="utf8")
            second_reference.write_text("b", encoding="utf8")

            written_paths = generate_profiles(
                model=FakeXTTSModel(),
                reference_files=[first_reference, second_reference],
                output_dir=output_dir,
                weights_name="arabic_xtts",
                overwrite=False,
            )

            self.assertEqual([path.name for path in written_paths], ["voice_one.json", "voice_one_2.json"])
            store = SpeakerProfileStore(output_dir)
            first_profile = store.get_profile("voice_one")
            self.assertIsNotNone(first_profile)
            self.assertEqual(first_profile["name"], "voice one")
            self.assertEqual(first_profile["description"], "Generated from voice one.wav using weights 'arabic_xtts'.")
            self.assertEqual(first_profile["gpt_cond_latent"][0][0], float(len("voice one.wav")))

    def test_main_uses_cli_args_instead_of_stale_globals(self):
        args = SimpleNamespace(
            weights=r".\tts_models\xtts-arabic-sa",
            references=r".\prompts\001_030.wav",
            weights_name="xtts-arabic-sa",
            output_root="speaker_profiles",
            device="cuda",
            overwrite=False,
        )
        fake_logger = FakeLogger()
        fake_model = object()

        with patch("generate_speaker_profiles.parse_args", return_value=args), \
             patch(
                 "generate_speaker_profiles.load_env_defaults",
                 return_value=SimpleNamespace(
                     custom_model_path="tts_models/base",
                     references="prompts/shahad.wav",
                     speaker_profiles_path="speaker_profiles/ignored",
                 ),
             ), \
             patch("generate_speaker_profiles.resolve_model_dir", return_value=Path("F:/repo/tts_models/xtts-arabic-sa")) as resolve_model_dir_mock, \
             patch("generate_speaker_profiles.derive_weights_name", return_value="xtts-arabic-sa"), \
             patch("generate_speaker_profiles.collect_reference_files", return_value=[Path("F:/repo/prompts/001_030.wav")]), \
             patch("generate_speaker_profiles.resolve_runtime_device", return_value=SimpleNamespace(type="cuda")), \
             patch("generate_speaker_profiles.load_xtts_model_from_path", return_value=fake_model), \
             patch("generate_speaker_profiles.generate_profiles", return_value=[Path("F:/repo/speaker_profiles/xtts-arabic-sa/001_030.json")]) as generate_profiles_mock, \
             patch("server.logging_utils.get_logger", return_value=fake_logger):
            main()

        resolve_model_dir_mock.assert_called_once_with(Path(r".\tts_models\xtts-arabic-sa"))
        self.assertEqual(generate_profiles_mock.call_args.kwargs["weights_name"], "xtts-arabic-sa")
        self.assertEqual(
            generate_profiles_mock.call_args.kwargs["output_dir"],
            Path("speaker_profiles").resolve() / "xtts-arabic-sa",
        )

    def test_main_uses_env_defaults_when_cli_omitted(self):
        args = SimpleNamespace(
            weights=None,
            references=None,
            weights_name=None,
            output_root=None,
            device="auto",
            overwrite=False,
        )
        env_defaults = SimpleNamespace(
            custom_model_path="tts_models/base",
            references="./prompts/shahad.wav",
            speaker_profiles_path="speaker_profiles/xtts-arabic-sa",
        )
        fake_logger = FakeLogger()
        fake_model = object()

        with patch("generate_speaker_profiles.parse_args", return_value=args), \
             patch("generate_speaker_profiles.load_env_defaults", return_value=env_defaults), \
             patch("generate_speaker_profiles.resolve_model_dir", return_value=Path("F:/repo/tts_models/base")) as resolve_model_dir_mock, \
             patch("generate_speaker_profiles.collect_reference_files", return_value=[Path("F:/repo/prompts/shahad.wav")]) as collect_reference_files_mock, \
             patch("generate_speaker_profiles.resolve_runtime_device", return_value=SimpleNamespace(type="cpu")), \
             patch("generate_speaker_profiles.load_xtts_model_from_path", return_value=fake_model), \
             patch("generate_speaker_profiles.generate_profiles", return_value=[Path("F:/repo/speaker_profiles/xtts-arabic-sa/shahad.json")]) as generate_profiles_mock, \
             patch("server.logging_utils.get_logger", return_value=fake_logger):
            main()

        resolve_model_dir_mock.assert_called_once_with((DEFAULT_PROJECT_ROOT / "tts_models/base").resolve())
        collect_reference_files_mock.assert_called_once_with((DEFAULT_PROJECT_ROOT / "prompts/shahad.wav").resolve())
        self.assertEqual(
            generate_profiles_mock.call_args.kwargs["output_dir"],
            (DEFAULT_PROJECT_ROOT / "speaker_profiles/xtts-arabic-sa").resolve(),
        )
        self.assertEqual(generate_profiles_mock.call_args.kwargs["weights_name"], "xtts-arabic-sa")

    def test_patch_xtts_audio_loader_falls_back_to_soundfile(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "voice.wav"
            sf.write(wav_path, [0.0, 0.5, -0.5, 0.0], 16000)

            def _raise_torchcodec(*_args, **_kwargs):
                raise ImportError("TorchCodec is required for load_with_torchcodec.")

            fake_module = SimpleNamespace(load_audio=_raise_torchcodec)
            patch_xtts_audio_loader(fake_module)

            audio = fake_module.load_audio(str(wav_path), 16000)

            self.assertEqual(tuple(audio.shape), (1, 4))
