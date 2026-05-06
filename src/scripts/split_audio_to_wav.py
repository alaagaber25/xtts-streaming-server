import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Edit these values before running the script.
INPUT_AUDIO_PATH = REPO_ROOT / "src/prompts/shahad.wav"
START_TIME = "00:00:01"
END_TIME = "00:00:07"
OUTPUT_WAV_PATH = REPO_ROOT / "src/prompts" / "shahad_1.wav"

OVERWRITE = True
AVOID_EXISTING_OUTPUT = True
TRIM_END_SILENCE = True
SILENCE_THRESHOLD_DB = -50.0
KEEP_TRAILING_SILENCE = 0.0

# Set SAMPLE_RATE to 24000 if you want XTTS-friendly prompt audio.
SAMPLE_RATE = 24000
MONO = True


def _parse_time(value: str | int | float) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into seconds."""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError("time value must be zero or greater")
        return seconds

    raw_value = value.strip()
    if not raw_value:
        raise ValueError("time value cannot be empty")

    parts = raw_value.split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = minutes * 60 + float(parts[1])
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = hours * 3600 + minutes * 60 + float(parts[2])
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"invalid time value {value!r}; use seconds, MM:SS, or HH:MM:SS"
        ) from exc

    if seconds < 0:
        raise ValueError("time value must be zero or greater")
    return seconds


def _format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise SystemExit(f"Could not find an available output filename near: {path}")


def _validate_config(start: float, end: float) -> tuple[Path, Path]:
    input_path = INPUT_AUDIO_PATH.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")

    if end <= start:
        raise SystemExit("END_TIME must be greater than START_TIME")

    if SAMPLE_RATE is not None and SAMPLE_RATE <= 0:
        raise SystemExit("SAMPLE_RATE must be greater than zero")

    if KEEP_TRAILING_SILENCE < 0:
        raise SystemExit("KEEP_TRAILING_SILENCE must be zero or greater")

    output_path = OUTPUT_WAV_PATH.expanduser().resolve()
    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")

    if output_path.exists() and AVOID_EXISTING_OUTPUT:
        output_path = _next_available_path(output_path)
    elif output_path.exists() and not OVERWRITE:
        raise SystemExit(
            f"Output file already exists: {output_path}\n"
            "Set OVERWRITE = True to replace it."
        )

    return input_path, output_path


def _probe_duration(input_path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe was not found on PATH.")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(f"Could not read input duration with ffprobe: {message}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise SystemExit(f"Could not parse input duration: {result.stdout!r}") from exc

    if duration <= 0:
        raise SystemExit(f"Input file has no readable audio duration: {input_path}")
    return duration


def _validate_time_window(
    start: float, end: float, duration: float, input_path: Path
) -> None:
    if start >= duration:
        raise SystemExit(
            f"START_TIME is outside the input audio.\n"
            f"Input: {input_path}\n"
            f"Input duration: {_format_seconds(duration)} seconds\n"
            f"START_TIME: {_format_seconds(start)} seconds"
        )

    if end > duration:
        raise SystemExit(
            f"END_TIME is past the input audio.\n"
            f"Input: {input_path}\n"
            f"Input duration: {_format_seconds(duration)} seconds\n"
            f"Requested window: {_format_seconds(start)} to {_format_seconds(end)} seconds"
        )


def _build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> list[str]:
    duration = end - start
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y" if OVERWRITE else "-n",
        "-ss",
        _format_seconds(start),
        "-i",
        str(input_path),
        "-t",
        _format_seconds(duration),
        "-map",
        "0:a:0",
        "-vn",
    ]

    if TRIM_END_SILENCE:
        silence_filter = (
            "areverse,"
            "silenceremove="
            "start_periods=1:"
            f"start_threshold={_format_seconds(SILENCE_THRESHOLD_DB)}dB:"
            f"start_silence={_format_seconds(KEEP_TRAILING_SILENCE)},"
            "areverse"
        )
        command.extend(["-af", silence_filter])

    if SAMPLE_RATE is not None:
        command.extend(["-ar", str(SAMPLE_RATE)])

    if MONO:
        command.extend(["-ac", "1"])

    command.extend(["-c:a", "pcm_s16le", str(output_path)])
    return command


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH.")

    start = _parse_time(START_TIME)
    end = _parse_time(END_TIME)
    input_path, output_path = _validate_config(start, end)
    input_duration = _probe_duration(input_path)
    _validate_time_window(start, end, input_duration, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = _build_ffmpeg_command(input_path, output_path, start, end)
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if not output_path.is_file() or output_path.stat().st_size <= 1024:
        raise SystemExit(
            f"ffmpeg did not create a usable WAV: {output_path}\n"
            "Check START_TIME, END_TIME, and the silence trimming settings."
        )

    print(f"Saved WAV: {output_path}", flush=True)


if __name__ == "__main__":
    main()
