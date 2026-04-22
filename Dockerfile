FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        sox \
        libsox-fmt-all \
        curl \
        wget \
        gcc \
        git \
        git-lfs \
        build-essential \
        libaio-dev \
        libsndfile1 \
        ssh \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_LINK_MODE=copy

WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY server ./server
RUN python -m pip install --upgrade pip uv && \
    uv sync --active --frozen --no-dev --no-editable && \
    python -m unidic download && \
    rm -rf /root/.cache /tmp/uv-cache


FROM python:3.11-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        sox \
        libsox-fmt-all \
        curl \
        libsndfile1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV NVIDIA_DISABLE_REQUIRE=0
ENV PYTHONUNBUFFERED=1
ENV NUM_THREADS=2

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY server ./server
RUN mkdir -p /app/tts_models

EXPOSE 80
CMD ["uvicorn", "server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "80"]
