from typing import Any, Callable, Iterator

import torch


def build_isolated_xtts_stream(
    *,
    model: Any,
    device: torch.device,
    text: str,
    language: str,
    gpt_cond_latent: torch.Tensor,
    speaker_embedding: torch.Tensor,
    stream_chunk_size: int,
    enable_text_splitting: bool,
    split_sentence_fn: Callable[[str, str, int], list[str]],
) -> Iterator[Any]:
    gpt_inference = getattr(getattr(model, "gpt", None), "gpt_inference", None)
    if gpt_inference is None or not hasattr(gpt_inference, "store_prefix_emb"):
        return model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=stream_chunk_size,
            enable_text_splitting=enable_text_splitting,
        )

    language = language.split("-", 1)[0]
    if enable_text_splitting:
        text_segments = split_sentence_fn(text, language, model.tokenizer.char_limits[language])
    else:
        text_segments = [text]

    def isolated_stream() -> Iterator[Any]:
        with torch.inference_mode():
            for text_segment in text_segments:
                normalized_segment = text_segment.strip().lower()
                text_tokens = torch.IntTensor(model.tokenizer.encode(normalized_segment, lang=language)).unsqueeze(0)
                text_tokens = text_tokens.to(device)

                if text_tokens.shape[-1] >= model.args.gpt_max_text_tokens:
                    raise ValueError("XTTS can only generate text with a maximum of 400 tokens.")

                fake_inputs = model.gpt.compute_embeddings(gpt_cond_latent, text_tokens)
                prefix_emb = gpt_inference.cached_prefix_emb
                gpt_generator = model.gpt.get_generator(
                    fake_inputs=fake_inputs,
                    top_k=50,
                    top_p=0.85,
                    temperature=0.75,
                    do_sample=True,
                    num_beams=1,
                    num_return_sequences=1,
                    length_penalty=1.0,
                    repetition_penalty=10.0,
                    output_attentions=False,
                    output_hidden_states=True,
                )

                last_tokens = []
                all_latents = []
                wav_gen_prev = None
                wav_overlap = None
                is_end = False

                while not is_end:
                    try:
                        # XTTS stores prefix embeddings on the shared GPT inference
                        # module, so restore the request-local prefix before each step.
                        gpt_inference.store_prefix_emb(prefix_emb)
                        x, latent = next(gpt_generator)
                        last_tokens.append(x)
                        all_latents.append(latent)
                    except StopIteration:
                        is_end = True

                    if is_end or len(last_tokens) >= stream_chunk_size:
                        gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                        wav_gen = model.hifigan_decoder(gpt_latents, g=speaker_embedding)
                        wav_chunk, wav_gen_prev, wav_overlap = model.handle_chunks(
                            wav_gen.squeeze(),
                            wav_gen_prev,
                            wav_overlap,
                            1024,
                        )
                        last_tokens = []
                        yield wav_chunk

    return isolated_stream()
