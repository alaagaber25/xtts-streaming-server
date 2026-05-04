import json
import wave
from datetime import datetime, timezone
from pathlib import Path


class TTSOutputRecorder:
    def __init__(
        self,
        output_root: Path,
        request_id: str,
        metadata: dict,
        sample_rate: int = 24000,
        sample_width: int = 2,
        channels: int = 1,
    ):
        self.request_id = request_id
        self.request_dir = output_root / request_id
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.chunk_count = 0
        self.total_bytes = 0
        self._final_wav = None

        self.request_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.request_dir / "metadata.json"
        self._metadata = {
            **metadata,
            "request_id": request_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "sample_rate": sample_rate,
            "sample_width": sample_width,
            "channels": channels,
            "status": "running",
        }
        self._write_metadata()

        self._final_wav = self._open_wav(self.request_dir / "final.wav")

    def write_chunk(self, pcm_bytes: bytes) -> None:
        chunk_path = self.request_dir / f"chunk_{self.chunk_count:04d}.wav"
        with self._open_wav(chunk_path) as chunk_wav:
            chunk_wav.writeframes(pcm_bytes)

        self._final_wav.writeframes(pcm_bytes)
        self.chunk_count += 1
        self.total_bytes += len(pcm_bytes)

    def close(self, status: str = "complete") -> None:
        if self._final_wav is not None:
            self._final_wav.close()
            self._final_wav = None

        self._metadata.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": self.chunk_count,
                "total_bytes": self.total_bytes,
                "final_wav": "final.wav",
                "status": status,
            }
        )
        self._write_metadata()

    def _open_wav(self, path: Path):
        wav_file = wave.open(str(path), "wb")
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(self.sample_width)
        wav_file.setframerate(self.sample_rate)
        return wav_file

    def _write_metadata(self) -> None:
        self._metadata_path.write_text(
            json.dumps(self._metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
