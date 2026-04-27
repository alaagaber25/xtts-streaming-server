FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y sox libsox-fmt-all curl wget gcc git git-lfs build-essential libaio-dev libsndfile1 ssh ffmpeg && \
    apt-get clean && apt-get -y autoremove

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install -r requirements.txt \
    && python -m pip cache purge

RUN python -m unidic download
RUN mkdir -p /app/src /app/tts_models /app/speaker_profiles

COPY .env /app/.env
COPY src /app/src
ENV NVIDIA_DISABLE_REQUIRE=1

ENV NUM_THREADS=2
WORKDIR /app/src
EXPOSE 8004
CMD ["python", "main.py"]
