import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.streaming import build_isolated_xtts_stream


class _FakeTokenizer:
    char_limits = {"en": 400}

    def encode(self, text: str, lang: str):
        return [1, 2, 3]


class _FakeGPTInference:
    def __init__(self) -> None:
        self.cached_prefix_emb = None

    def store_prefix_emb(self, prefix_emb) -> None:
        self.cached_prefix_emb = prefix_emb


class _FakeGPT:
    def __init__(self) -> None:
        self.gpt_inference = _FakeGPTInference()

    def compute_embeddings(self, cond_latents, text_tokens):
        self.gpt_inference.store_prefix_emb(cond_latents)
        return torch.ones((1, text_tokens.shape[1] + 1), dtype=torch.long)

    def get_generator(self, fake_inputs, **kwargs):
        def iterator():
            for _ in range(2):
                prefix_value = float(self.gpt_inference.cached_prefix_emb.flatten()[0].item())
                yield torch.tensor([prefix_value]), torch.tensor([prefix_value], dtype=torch.float32)

        return iterator()


class _FakeModel:
    def __init__(self) -> None:
        self.gpt = _FakeGPT()
        self.tokenizer = _FakeTokenizer()
        self.args = SimpleNamespace(gpt_max_text_tokens=400)

    def hifigan_decoder(self, gpt_latents, g):
        return gpt_latents

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        if wav_gen.ndim == 0:
            wav_gen = wav_gen.unsqueeze(0)
        return wav_gen[-1:], wav_gen, None


class StreamingHelperTests(unittest.TestCase):
    def test_interleaved_streams_keep_request_specific_prefix_state(self):
        model = _FakeModel()
        split_sentence_fn = lambda text, language, limit: [text]

        iterator_a = build_isolated_xtts_stream(
            model=model,
            device=torch.device("cpu"),
            text="alpha",
            language="en",
            gpt_cond_latent=torch.tensor([[[1.0] * 1024]], dtype=torch.float32),
            speaker_embedding=torch.tensor([[[0.5]]], dtype=torch.float32),
            stream_chunk_size=1,
            enable_text_splitting=True,
            split_sentence_fn=split_sentence_fn,
        )
        iterator_b = build_isolated_xtts_stream(
            model=model,
            device=torch.device("cpu"),
            text="beta",
            language="en",
            gpt_cond_latent=torch.tensor([[[2.0] * 1024]], dtype=torch.float32),
            speaker_embedding=torch.tensor([[[0.5]]], dtype=torch.float32),
            stream_chunk_size=1,
            enable_text_splitting=True,
            split_sentence_fn=split_sentence_fn,
        )

        observed = [
            float(next(iterator_a).item()),
            float(next(iterator_b).item()),
            float(next(iterator_a).item()),
            float(next(iterator_b).item()),
        ]

        self.assertEqual(observed, [1.0, 2.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
