# XTTS Streaming Server

Lightweight FastAPI server for XTTS streaming inference with:

- explicit custom checkpoint selection via `CUSTOM_MODEL_PATH`
- external speaker profile loading via `SPEAKER_PROFILES_PATH`
- `speaker_profile_id` support in `/tts` and `/tts_stream`
- a modular server layout that is easier to extend and maintain

This is still a demo-style XTTS server. It is useful for local integration and experimentation, but it is not designed for high-concurrency production traffic.

## What Changed

Compared with the original upstream layout, this codebase now:

- loads configuration from `.env` for local runs
- supports choosing one exact checkpoint directory from a folder of multiple models
- supports external speaker profiles from disk
- accepts either:
  - `speaker_profile_id`, or
  - raw `speaker_embedding` + `gpt_cond_latent`
- splits the server into focused modules instead of keeping everything in one file

## Project Layout

The source code lives under [`src/`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src>):

- [`src/core/config.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/core/config.py>)  
  environment loading, device selection, core settings
- [`src/core/patches.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/core/patches.py>)  
  compatibility patch for Hugging Face generation
- [`src/models/xtts_loader.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/models/xtts_loader.py>)  
  XTTS config/model loading and default-model download fallback
- [`src/speakers/loader.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/speakers/loader.py>)  
  external and model-native speaker discovery
- [`src/speakers/resolver.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/speakers/resolver.py>)  
  `speaker_profile_id` lookup and latent tensor construction
- [`src/schemas/requests.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/schemas/requests.py>)  
  request models for `/tts` and `/tts_stream`
- [`src/audio/processing.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/audio/processing.py>)  
  audio post-processing helpers
- [`src/routers/tts.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/routers/tts.py>)  
  `/tts` and `/tts_stream`
- [`src/routers/speakers.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/routers/speakers.py>)  
  `/clone_speaker` and `/studio_speakers`
- [`src/routers/languages.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/routers/languages.py>)  
  `/languages`
- [`src/app/main.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/app/main.py>)  
  thin FastAPI entrypoint

[`src/main.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/src/main.py>) remains as the runnable entrypoint.

## Requirements

- Docker with NVIDIA GPU support for CUDA images
- acceptance of the Coqui CPML license

Setting `COQUI_TOS_AGREED=1` means you have read and agreed to the [CPML license](https://coqui.ai/cpml).

## Checkpoint Layout

`CUSTOM_MODEL_PATH` must point to one exact XTTS model directory.

Required files:

```text
my_model/
  config.json
  model.pth
  vocab.json
```

Optional extra files may also be present, for example:

```text
my_model/
  config.json
  model.pth
  vocab.json
  dvae.pth
  mel_stats.pth
  speakers_xtts.pth
```

If `CUSTOM_MODEL_PATH` is set and does not point to a directory containing at least `config.json` and `model.pth`, the server now fails fast with a clear error.

If `CUSTOM_MODEL_PATH` is unset or empty, the server downloads and uses the default:

```text
tts_models/multilingual/multi-dataset/xtts_v2
```

## Speaker Profile Layout

External speaker profiles live under `SPEAKER_PROFILES_PATH`.

Each profile directory should contain:

```text
speaker_profiles/
  sa/
    fahad/
      speaker_embedding.json
      gpt_cond_latent.json
```

The server exposes both the full path-like id and a leaf alias when there is no conflict.  
In the example above, both of these may work:

- `sa/fahad`
- `fahad`

## Configuration

Local runs load [`.env`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/.env>) automatically through `python-dotenv`.

Example:

```env
COQUI_TOS_AGREED=1
NUM_THREADS=2
USE_DEEPSPEED=0
CUSTOM_MODEL_PATH=tts_models/sa
SPEAKER_PROFILES_PATH=speaker_profiles
SAVE_TTS_OUTPUTS=0
TTS_OUTPUTS_PATH=tts_outputs
XTTS_PORT=8004
TTS_TEMPERATURE=0.35
TTS_TOP_K=5
TTS_TOP_P=0.6
TTS_LENGTH_PENALTY=1.5
TTS_REPETITION_PENALTY=10.0
TTS_DO_SAMPLE=1
TTS_SEED=6
```

Main settings:

- `COQUI_TOS_AGREED`  
  required for Coqui model usage
- `NUM_THREADS`  
  torch CPU thread count
- `USE_DEEPSPEED`  
  optional, off by default
- `USE_CPU`  
  set to `1` to force CPU mode
- `CUSTOM_MODEL_PATH`  
  exact model directory to load
- `SPEAKER_PROFILES_PATH`  
  root directory containing external speaker profiles
- `SAVE_TTS_OUTPUTS`  
  set to `1` to save `/tts_stream` output for server-side listening/debugging
- `TTS_OUTPUTS_PATH`  
  directory where saved request audio is written when `SAVE_TTS_OUTPUTS=1`
- `TTS_TEMPERATURE`, `TTS_TOP_K`, `TTS_TOP_P`  
  XTTS GPT sampling controls; lower-variance values reduce hallucinated continuation
- `TTS_LENGTH_PENALTY`  
  generation length setting; with `num_beams=1`, Transformers warns this is effectively unused
- `TTS_REPETITION_PENALTY`  
  repetition control passed to XTTS generation
- `TTS_DO_SAMPLE`  
  keep this enabled for streaming; `do_sample=False` currently fails in this XTTS streaming stack
- `TTS_SEED`  
  optional fixed random seed for more repeatable sampling; `6` tested well with the low-variance preset

## Important Docker Note

The server reads `.env` only if that file exists inside the runtime environment.

That means:

- local Python runs pick up `.env`
- Docker containers do **not** see `.env` automatically unless you pass it with `--env-file`

So for Docker, use:

```powershell
docker run --gpus=all --env-file .env ...
```

## Run with Docker

### Build

From the repo root:

```powershell
docker build -t xtts-stream -f Dockerfile.cuda121 .
```

Available Dockerfiles:

- `Dockerfile.cuda121`
- `Dockerfile`
- `Dockerfile.cpu`

### Run with a custom checkpoint and speaker profiles

This is the recommended pattern when your repo contains multiple checkpoint folders:

```powershell
docker run --gpus=all `
  --env-file .env `
  -v F:\VOOM-AI\GitHubs\TTS\xtts-server\xtts\tts_models:/app/tts_models `
  -v F:\VOOM-AI\GitHubs\TTS\xtts-server\xtts\speaker_profiles:/app/speaker_profiles `
  -v F:\VOOM-AI\GitHubs\TTS\xtts-server\xtts\tts_outputs:/app/tts_outputs `
  -p 8004:8004 `
  xtts-stream
```

With this setup, `CUSTOM_MODEL_PATH` in `.env` chooses the exact checkpoint directory, for example:

```env
  CUSTOM_MODEL_PATH=tts_models/sa
```

If `SAVE_TTS_OUTPUTS=1`, each streaming request is saved under `TTS_OUTPUTS_PATH`.
For Docker, keep `TTS_OUTPUTS_PATH=tts_outputs` and mount your host folder to
`/app/tts_outputs` as shown above. The files then appear on Windows under
`F:\VOOM-AI\GitHubs\TTS\xtts-server\xtts\tts_outputs`, not only inside the
container.

Each saved request folder contains:

```text
tts_outputs/
  20260503_153012_9f4a2b1c/
    metadata.json
    final.wav
    chunk_0000.wav
    chunk_0001.wav
```

### Run on CPU

```powershell
docker build -t xtts-stream-cpu -f Dockerfile.cpu .

docker run `
  --env-file .env `
  -e USE_CPU=1 `
  -v F:\VOOM-AI\GitHubs\TTS\xtts-server\xtts\tts_models:/app/tts_models `
  -v F:\VOOM-AI\GitHubs\TTS\xtts-server\xtts\speaker_profiles:/app/speaker_profiles `
  -v F:\VOOM-AI\GitHubs\TTS\xtts-server\xtts\tts_outputs:/app/tts_outputs `
  -p 8004:8004 `
  xtts-stream-cpu
```

## Run Locally Without Docker

From the repo root:

```powershell
python -m pip install -r requirements.txt
python src/main.py
```

For local runs, `.env` is read from the project root and relative paths are resolved from the repo root.

Example local `.env` values:

```env
CUSTOM_MODEL_PATH=tts_models/sa
SPEAKER_PROFILES_PATH=speaker_profiles
XTTS_PORT=8004
```

## API

### `GET /languages`

Returns the languages available in the loaded XTTS config.

### `GET /studio_speakers`

Returns model-native speakers and external speaker profiles together in one object.

### `POST /clone_speaker`

Accepts a reference WAV file and returns:

- `speaker_embedding`
- `gpt_cond_latent`

This is useful when you want to generate a new speaker profile from audio.

### `POST /tts_stream`

Streaming synthesis endpoint.

It accepts either:

#### Option A: server-managed speaker profile

```json
{
  "text": "السلام عليكم",
  "language": "ar",
  "speaker_profile_id": "fahad",
  "add_wav_header": false,
  "stream_chunk_size": "20"
}
```

#### Option B: raw latents

```json
{
  "text": "السلام عليكم",
  "language": "ar",
  "speaker_embedding": [0.1, 0.2],
  "gpt_cond_latent": [[0.1, 0.2]],
  "add_wav_header": false,
  "stream_chunk_size": "20"
}
```

### `POST /tts`

Non-streaming synthesis endpoint with the same speaker input options as `/tts_stream`.

## Testing

There is a simple smoke test in [`test/test_streaming.py`](</f:/VOOM-AI/GitHubs/xtts-server/xtts-original-streaming/test/test_streaming.py>).

From the repo root:

```powershell
python -m pip install requests
cd test
python test_streaming.py --server_url http://localhost:8004 --output_file smoke.wav
```

Use a custom reference WAV:

```powershell
python test_streaming.py `
  --server_url http://localhost:8004 `
  --ref_file F:\path\to\reference.wav `
  --output_file custom.wav
```

If `--output_file` is omitted, the script tries to play audio through `ffplay`.

## Switching Between Multiple Checkpoints

If your `tts_models` folder contains multiple checkpoint directories:

```text
tts_models/
  sa/
  eg/
  custom_v3/
```

then switch models by changing `CUSTOM_MODEL_PATH`:

```env
CUSTOM_MODEL_PATH=tts_models/sa
```

or:

```env
CUSTOM_MODEL_PATH=tts_models/eg
```

Then restart the server.

The model is loaded once at startup, so changing checkpoints always requires a restart.

## XTTS v2 Streaming Stability Notes

This project hit an XTTS v2 behavior where short text sometimes produced more audio than expected. The saved output folder could contain more `chunk_XXXX.wav` files than a human would expect from the text, and the later chunks could sound unclear, mumbled, or like gibberish.

The important finding is that these files are usually not recorder leftovers. In this server, `chunk_count` in `metadata.json` is incremented only when XTTS yields an audio chunk. If `metadata.json` says `chunk_count: 8`, then XTTS actually produced eight streaming audio chunks.

### Streaming Chunks Are Not Text Chunks

`stream_chunk_size` is an XTTS audio-token streaming setting. It does not mean words, sentence pieces, or text chunks.

For example:

```env
stream_chunk_size=25
```

means XTTS emits audio after roughly 25 generated audio tokens. In our saved files, one chunk was often about 1.1 seconds. So a short sentence can produce 3 chunks when the model stops cleanly, or 7-9 chunks when the autoregressive generator keeps going.

### Controlled Matrix Findings

We tested the same Arabic sentence across models, speaker profiles, sampling presets, and seeds:

```text
وبعدها كيف اقدر اساعدك اليوم
```

The unstable cases were real generation continuations. With the `new_sa` checkpoint, default sampling could over-generate with more than one speaker profile:

```text
new_sa + eg/saied + default seed4:       9 chunks, 9.845s
new_sa + eg/saied + default seed1:       8 chunks, 8.640s
new_sa + sa/nada  + conservative seed6:  8 chunks, 8.821s
new_sa + sa/nada  + default seed3:       7 chunks, 7.755s
new_sa + sa/shahad + default seed2:      7 chunks, 7.339s
```

So the issue was not only one bad profile. It is a combination of:

- checkpoint behavior
- speaker latents
- sampling randomness
- stop-token prediction
- text normalization and punctuation

The best tested preset was:

```env
TTS_TEMPERATURE=0.35
TTS_TOP_K=5
TTS_TOP_P=0.6
TTS_REPETITION_PENALTY=10.0
TTS_DO_SAMPLE=1
TTS_SEED=6
```

One especially good sample was:

```text
matrix_runs/direct_20260505_025450/new_sa_sa_shahad_low_variance_seed6/final.wav
```

That run used:

```text
temperature=0.35
top_k=5
top_p=0.6
repetition_penalty=10.0
seed=6
```

and produced:

```text
4 chunks, 3.669s
```

### Why Not `do_sample=False`

In this XTTS streaming stack, `do_sample=False` is not currently a safe fix. The controlled matrix showed streaming errors like:

```text
GPT2InferenceModel object has no attribute greedy_search
```

So the practical fix is not greedy decoding. The safer path is low-variance sampling with a fixed seed.

### Text Normalization And EOS Cues

A major contributor was normalization. The old normalization path stripped all punctuation, including:

```text
.
!
?
؟
```

That removed sentence-ending cues before text reached XTTS. XTTS v2 was trained with punctuation and uses these characters as signals for sentence boundaries and end-of-sequence prediction. If the model never sees a final `.`, `!`, or `?`, it is more likely to keep generating after the intended text.

The current normalization keeps those cues:

- `_fix_punctuation_spacing()` runs before character filtering. It removes space before punctuation, collapses repeated spaces after punctuation, and prevents terminal punctuation plus trailing-space patterns from reaching XTTS.
- `_ARABIC_TO_LATIN_PUNCT` maps Arabic question marks, especially `؟`, to `?` so question endings survive the filter.
- `_SENTENCE_ENDERS` allows `.`, `!`, and `?` through the character filter.
- `_ensure_terminal_punctuation()` runs last. If the final normalized text does not end in `.`, `!`, or `?`, it appends `.`.

Example effect:

```text
before: وبعدها كيف اقدر اساعدك اليوم
after:  وبعدها كيف اقدر اساعدك اليوم.
```

This gives XTTS a clear stop cue and reduces end-of-sentence hallucination.

### Speaker Latents And Model Pairing

Speaker profiles are not just identity labels; they are conditioning tensors. A profile generated against one checkpoint can behave differently with another checkpoint.

If a speaker starts producing long tails, mumbling, or unstable chunk counts:

- regenerate that speaker profile using the active checkpoint
- compare the same text with another speaker
- compare the same speaker on another checkpoint
- inspect `metadata.json` for `generation`, `seed`, `speaker_profile_id`, and `chunk_count`

### Related Upstream Reports

Similar XTTS v2 behavior has been reported upstream:

- [Coqui TTS issue #3964](https://github.com/coqui-ai/TTS/issues/3964): short utterances producing hallucinated audio at the end
- [Coqui TTS issue #3516](https://github.com/coqui-ai/TTS/issues/3516): fine-tuned XTTS v2 making strange sounds for short text
- [Coqui TTS discussion #4146](https://github.com/coqui-ai/TTS/discussions/4146): reducing end-of-sentence hallucinations and seed sensitivity
- [Hugging Face XTTS-v2 discussion #16](https://huggingface.co/coqui/XTTS-v2/discussions/16): some checkpoint versions producing bad voices or extra gibberish
- [Hugging Face XTTS-v2 discussion #104](https://huggingface.co/coqui/XTTS-v2/discussions/104): text formatting and punctuation influencing hallucination

## Fixes Applied In This Repo

- A single-request lock serializes `/tts`, `/tts_stream`, and `/clone_speaker` model access. XTTS streaming is not safe to run concurrently against one shared GPU model.
- Low-variance generation settings are configurable from `.env`.
- `TTS_SEED=6` can be used to make sampling more repeatable.
- Streaming request metadata records generation settings and seed.
- Text normalization preserves sentence-ending punctuation and guarantees terminal punctuation.
- `matrix_runs/` is ignored by Docker builds so diagnostic WAVs do not bloat the build context.

## Troubleshooting

### The server downloads `xtts_v2` instead of my checkpoint

Usually one of these is true:

- `CUSTOM_MODEL_PATH` is unset or empty
- Docker was started without `--env-file .env`
- `CUSTOM_MODEL_PATH` points to the wrong directory
- the target directory does not contain both `config.json` and `model.pth`

### The server sees speaker profiles but not the custom model

That usually means your speaker profile volume mount is correct, but your model path is not.

Check that:

- your model files are mounted into the container
- `CUSTOM_MODEL_PATH` matches the container path, not the Windows host path

### `422 Unprocessable Entity` on `/tts_stream`

This usually means the request body is missing either:

- `speaker_profile_id`, or
- both `speaker_embedding` and `gpt_cond_latent`

### `speaker_profile_id` is not found

Check:

- the profile directory exists under `SPEAKER_PROFILES_PATH`
- it contains both `speaker_embedding.json` and `gpt_cond_latent.json`
- the server was restarted after adding the profile
- the id returned by `/studio_speakers` matches what the client is using

### Extra `chunk_XXXX.wav` files or gibberish at the end

Check `metadata.json` first:

- if `chunk_count` matches the number of chunk files, XTTS generated those chunks
- if the final chunks sound like mumbling or nonsense, the model likely over-generated
- confirm the text ends with `.`, `!`, or `?` after normalization
- use the low-variance `.env` settings listed above
- keep `TTS_DO_SAMPLE=1`; do not switch streaming to `do_sample=False`
- try `TTS_SEED=6`
- regenerate the speaker profile with the active checkpoint
- compare against another speaker profile and another checkpoint

### Saved request folder says `status: running`

This can happen if the HTTP client disconnects or the request is killed while a stream is still open. In that case, `final.wav` may be incomplete or empty. Rerun the request and prefer completed folders with:

```json
"status": "complete"
```

## Notes

- this server serializes model access and is best for one active streaming request at a time
- loading a different checkpoint requires restart
- loading new external speaker profiles also requires restart
- model-native speakers and external profiles are merged at runtime for lookup
