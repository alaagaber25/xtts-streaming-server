import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests


TEST_DIR = Path(__file__).resolve().parent


def load_speaker(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_request(
    index: int,
    server_url: str,
    speaker: Dict[str, object],
    language: str,
    stream_chunk_size: int,
    text_template: str,
    save_dir: Path | None,
) -> Dict[str, object]:
    payload = dict(speaker)
    payload["text"] = text_template.format(index=index)
    payload["language"] = language
    payload["stream_chunk_size"] = stream_chunk_size

    output_path = None
    output_handle = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"request-{index}.wav"
        output_handle = output_path.open("wb")

    start = time.perf_counter()
    first_chunk_latency = None
    bytes_received = 0

    try:
        with requests.post(
            f"{server_url}/tts_stream",
            json=payload,
            stream=True,
            timeout=(10, 300),
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024):
                if not chunk:
                    continue

                if first_chunk_latency is None:
                    first_chunk_latency = time.perf_counter() - start

                bytes_received += len(chunk)
                if output_handle is not None:
                    output_handle.write(chunk)
    finally:
        if output_handle is not None:
            output_handle.close()

    total_time = time.perf_counter() - start
    if first_chunk_latency is None:
        raise RuntimeError(f"request {index} returned no audio bytes")

    return {
        "index": index,
        "first_chunk_latency": first_chunk_latency,
        "total_time": total_time,
        "bytes_received": bytes_received,
        "output_path": str(output_path) if output_path is not None else None,
    }


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile for an empty list")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="http://localhost:8002")
    parser.add_argument("--speaker_json", default=str(TEST_DIR / "default_speaker.json"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--stream_chunk_size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--text_template",
        default="Concurrent XTTS request {index}. This is a scheduler smoke test.",
    )
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    speaker = load_speaker(Path(args.speaker_json))
    save_dir = Path(args.save_dir) if args.save_dir else None

    print(
        f"Running {args.concurrency} concurrent requests against {args.server_url}/tts_stream",
        flush=True,
    )

    started = time.perf_counter()
    results: List[Dict[str, object]] = []
    errors: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_request,
                index,
                args.server_url,
                speaker,
                args.language,
                args.stream_chunk_size,
                args.text_template,
                save_dir,
            )
            for index in range(args.concurrency)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(
                    "request {index}: first_chunk={first_chunk_latency:.3f}s total={total_time:.3f}s bytes={bytes_received}".format(
                        **result
                    ),
                    flush=True,
                )
            except Exception as exc:
                errors.append(str(exc))
                print(f"request failed: {exc}", file=sys.stderr, flush=True)

    wall_time = time.perf_counter() - started
    if errors:
        print(f"{len(errors)} request(s) failed.", file=sys.stderr, flush=True)
        for error in errors:
            print(error, file=sys.stderr, flush=True)
        return 1

    first_chunk_latencies = [float(result["first_chunk_latency"]) for result in results]
    total_times = [float(result["total_time"]) for result in results]

    print("", flush=True)
    print(f"Completed in {wall_time:.3f}s wall time", flush=True)
    print(f"mean first chunk: {statistics.mean(first_chunk_latencies):.3f}s", flush=True)
    print(f"p95 first chunk: {percentile(first_chunk_latencies, 0.95):.3f}s", flush=True)
    print(f"mean total time: {statistics.mean(total_times):.3f}s", flush=True)
    print(f"p95 total time: {percentile(total_times, 0.95):.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
