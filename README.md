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
  -v F:\VOOM-AI\GitHubs\xtts-server\xtts\tts_models:/app/tts_models `
  -v F:\VOOM-AI\GitHubs\xtts-server\xtts\speaker_profiles:/app/speaker_profiles `
  -v F:\VOOM-AI\GitHubs\xtts-server\xtts\tts_outputs:/app/tts_outputs `
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
`F:\VOOM-AI\GitHubs\xtts-server\xtts\tts_outputs`, not only inside the
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
  -v F:\VOOM-AI\GitHubs\xtts-server\xtts\tts_models:/app/tts_models `
  -v F:\VOOM-AI\GitHubs\xtts-server\xtts\speaker_profiles:/app/speaker_profiles `
  -v F:\VOOM-AI\GitHubs\xtts-server\xtts\tts_outputs:/app/tts_outputs `
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

## Notes

- this server is best for one active streaming request at a time
- loading a different checkpoint requires restart
- loading new external speaker profiles also requires restart
- model-native speakers and external profiles are merged at runtime for lookup
