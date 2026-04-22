# XTTS streaming server
This fork keeps Coqui's low-latency XTTS streaming path, but wraps it in a production-oriented scheduler:

`Client -> FastAPI endpoint -> request queue -> micro-batch scheduler -> GPU worker -> streaming dispatcher -> client`

The original upstream server explicitly warns that it does not support concurrent streaming requests and is intended as a demo. This fork adds queue-based request isolation, round-robin stream advancement, and batch-window scheduling while keeping the `/tts_stream` API contract intact.

## Recommended runtime
Use Docker on Linux containers, including Docker Desktop with WSL2 on Windows. Running the server directly in a Windows virtualenv is fragile because the legacy XTTS streaming stack is sensitive to `torch` and `transformers` version drift.

Pinned runtime in this repo:
- Python 3.11
- CUDA 12.1 PyTorch wheels resolved from `uv.lock`
- `transformers==4.41.2`
- single Uvicorn worker
- single XTTS GPU worker with small micro-batches

The dependency source of truth lives in [`pyproject.toml`](pyproject.toml). Docker, local server installs, and the client scripts all build from the optional dependency groups defined there.

https://github.com/coqui-ai/xtts-streaming-server/assets/17219561/7220442a-e88a-4288-8a73-608c4b39d06c


## 1) Run the server

### Use a pre-built image

CUDA 12.1:

```bash
$ docker run --gpus=all -e COQUI_TOS_AGREED=1 --rm -p 8000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest-cuda121
```

Run with a fine-tuned model:

Make sure the model folder `/path/to/model/folder`  contains the following files:
- `config.json`
- `model.pth`
- `vocab.json`

```bash
$ docker run -v /path/to/model/folder:/app/tts_models --gpus=all -e COQUI_TOS_AGREED=1  --rm -p 8000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest`
```

Setting the `COQUI_TOS_AGREED` environment variable to `1` indicates you have read and agreed to
the terms of the [CPML license](https://coqui.ai/cpml). (Fine-tuned XTTS models also are under the [CPML license](https://coqui.ai/cpml))

### Build and run this fork

Copy `.env.example` to `.env` and set your runtime values such as:
- `DEFAULT_SPEAKER_PROFILE_ID`
- `XTTS_PORT`
- `BATCH_COLLECTION_WINDOW`
- `MAX_BATCH_SIZE`
- `GPU_WORKER_COUNT`

Then run the server with one command:

```bash
$ cd xtts-streaming-server
$ docker compose up --build
```

By default this fork tries `http://localhost:8002` to avoid collisions with other local services already using port `8000`. You can override the port explicitly with `XTTS_PORT`.

The production compose setup is GPU-only and declares `gpus: all` directly in [`docker-compose.yml`](docker-compose.yml). It persists the downloaded XTTS model cache in a Docker volume and mounts `./tts_models` and `./speaker_profiles` into the container so custom models and saved speaker profiles survive restarts.

### Local run without Docker

Docker is still the recommended production path, but running locally is useful for development and debugging.
GPU is recommended when available because it most closely matches the container runtime.

Install the same core system packages used by the Docker image:
- `ffmpeg`
- `sox`
- `libsndfile`
- `git`
- `git-lfs`
- a C/C++ build toolchain such as `build-essential` on Linux

Create and activate a virtual environment:

```bash
$ cd xtts-streaming-server
$ python -m venv .venv
$ source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the server dependencies from [`pyproject.toml`](pyproject.toml):

```bash
$ uv sync
$ uv run python -m unidic download
```

If you prefer `uv pip`, use:

```bash
$ uv pip install .
$ python -m unidic download
```

If XTTS fails on Windows with `WinError 127` while importing `torchaudio`, check that `torch` and `torchaudio` are the same release. A mismatched pair such as `torch==2.5.1+cu121` with `torchaudio==2.11.0` will fail before the model loads. Repair it with:

```powershell
python -m pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu121 torchaudio==2.5.1+cu121
```

Copy `.env.example` to `.env` and set any runtime values you need. The same `.env` file used by Docker is read for local runs too.

Important environment variables:
- `COQUI_TOS_AGREED=1`
- `CUSTOM_MODEL_PATH`
- `SPEAKER_PROFILES_PATH`
- `DEFAULT_SPEAKER_PROFILE_ID`
- `NUM_THREADS`
- `BATCH_COLLECTION_WINDOW`
- `MAX_BATCH_SIZE`
- `QUEUE_POLL_INTERVAL`
- `GPU_WORKER_COUNT`

Start the server directly with Uvicorn:

```bash id="local-run"
$ uv run uvicorn server.app:create_app --factory --host 0.0.0.0 --port 8002
```

If you already activated `.venv`, `uvicorn server.app:create_app --factory --host 0.0.0.0 --port 8002` should also work, but `uv run ...` is the most reliable option because it guarantees the correct interpreter and lockfile-resolved environment are selected.

Setting the `COQUI_TOS_AGREED` environment variable to `1` indicates you have read and agreed to
the terms of the [CPML license](https://coqui.ai/cpml). (Fine-tuned XTTS models also are under the [CPML license](https://coqui.ai/cpml))

## 2) Server-managed speaker profiles

This fork can now save cloned voices on the server so clients can refer to them by `speaker_profile_id` instead of resending latents every time.

Clone and save a profile:

```bash
$ curl -X POST http://localhost:8002/clone_speaker \
    -F "wav_file=@reference.wav" \
    -F "speaker_profile_id=my_voice" \
    -F "name=My Voice"
```

List saved profiles:

```bash
$ curl http://localhost:8002/speaker_profiles
```

Use a saved profile for streaming:

```bash
$ curl -X POST http://localhost:8002/tts_stream \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"Hello there\",\"language\":\"en\",\"speaker_profile_id\":\"my_voice\"}"
```

`/tts` and `/tts_stream` still accept raw `speaker_embedding` and `gpt_cond_latent`, so existing clients do not have to change immediately.

## 3) Testing the running server

Once your Docker container is running, you can test that it's working properly. You will need to run the following code from a fresh terminal.

### Clone `xtts-streaming-server` if you haven't already

```bash
$ git clone git@github.com:coqui-ai/xtts-streaming-server.git
```

### Install a lightweight client environment

```bash
$ cd xtts-streaming-server
$ python -m pip install ".[client]"
```

### Using the single-request test script

```bash
$ cd xtts-streaming-server/test
$ python test_streaming.py --server_url http://localhost:8002 --output_file smoke.wav
```

### Using the concurrency test script

```bash
$ cd xtts-streaming-server/test
$ python test_concurrency.py --server_url http://localhost:8002 --concurrency 2
```

The concurrency script reports:
- time to first chunk per request
- total request duration per request
- aggregate mean and p95 latency

### Health check

```bash
$ curl http://localhost:8002/health
```

The response includes the active device, queue depth, open session count, and scheduler settings.
