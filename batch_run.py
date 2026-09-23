"""Reproducible local batch runner. Meeting data remains in the chosen output folder."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from document_export import make_docx
from pipeline import diarize, extract, lines_from_segments, transcribe

STAGES = ("asr", "diarization", "extraction", "docx")


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required prior-stage file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_lines(asr_segments: list[dict], diar_lines: list[dict]) -> list[dict]:
    """Keep ASR sentences intact for actions; attach only supported cluster labels."""
    lines = lines_from_segments(asr_segments)
    for line in lines:
        scores = {}
        for turn in diar_lines:
            overlap = max(0.0, min(line["end"], turn["end"]) - max(line["start"], turn["start"]))
            if overlap and turn["speaker"] != "Не определён":
                scores[turn["speaker"]] = scores.get(turn["speaker"], 0.0) + overlap
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        length = max(0.0, line["end"] - line["start"])
        if ranked and length and ranked[0][1] >= 0.75 * length and (len(ranked) == 1 or ranked[1][1] <= 0.15 * length):
            line["speaker"] = ranked[0][0]
            line["needs_review"] = False
        else:
            line["speaker"] = "Не определён"
            line["needs_review"] = True
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recording analysis on local models")
    parser.add_argument("--audio", type=Path, help="Local recording path; required for ASR/diarization")
    parser.add_argument("--output", required=True, type=Path, help="Local output directory")
    parser.add_argument("--from-stage", choices=STAGES, default="asr")
    parser.add_argument("--stop-after", choices=STAGES, default="docx")
    parser.add_argument("--review-note", action="append", default=[],
                        help="Optional uncertainty to display; never changes transcript words")
    args = parser.parse_args()
    start, stop = STAGES.index(args.from_stage), STAGES.index(args.stop_after)
    if start > stop:
        parser.error("--from-stage must be at or before --stop-after")
    if start <= STAGES.index("diarization") and not args.audio:
        parser.error("--audio is required for ASR or diarization")
    if args.audio and not args.audio.is_file():
        parser.error(f"Recording does not exist: {args.audio}")
    args.output.mkdir(parents=True, exist_ok=True)
    timings_path = args.output / "timings.json"
    timings = read_json(timings_path) if timings_path.exists() else {}

    if start <= 0 <= stop:
        t = time.perf_counter()
        segments = transcribe(args.audio)
        timings["asr"] = time.perf_counter() - t
        write_json(args.output / "asr.json", segments)
        write_json(timings_path, timings)
        print(f"ASR {timings['asr']:.2f}s, {len(segments)} segments", flush=True)
    if start <= 1 <= stop:
        segments = read_json(args.output / "asr.json")
        t = time.perf_counter()
        lines = lines_from_segments(diarize(args.audio, segments))
        timings["diarization"] = time.perf_counter() - t
        write_json(args.output / "transcript.json", lines)
        write_json(timings_path, timings)
        print(f"Diarization {timings['diarization']:.2f}s, {len(lines)} lines", flush=True)
    if start <= 2 <= stop:
        diar_lines = read_json(args.output / "transcript.json")
        asr_segments = read_json(args.output / "asr.json")
        lines = source_lines(asr_segments, diar_lines)
        write_json(args.output / "source_transcript.json", lines)
        result = extract(lines, review_notes=args.review_note)
        timings.update(result.get("stage_seconds", {}))
        write_json(args.output / "result.json", result)
        write_json(timings_path, timings)
        print(f"Extraction {timings.get('action_extraction', 0):.2f}s, "
              f"summary {timings.get('summary_generation', 0):.2f}s, "
              f"{len(result['actions'])} actions", flush=True)
    if start <= 3 <= stop:
        lines = read_json(args.output / "source_transcript.json")
        diar_lines = read_json(args.output / "transcript.json")
        result = read_json(args.output / "result.json")
        t = time.perf_counter()
        title = "Протокол совещания"
        if args.audio:
            stem = args.audio.stem
            title = ("Протокол совещания" + stem[len("Совещание"):]
                     if stem.casefold().startswith("совещание") else f"Протокол {stem}")
        data = make_docx(lines, result, result["actions"], title=title,
                         diarized_lines=diar_lines)
        (args.output / "protocol.docx").write_bytes(data)
        timings["docx_export"] = time.perf_counter() - t
        write_json(timings_path, timings)
        print(f"DOCX {timings['docx_export']:.2f}s, {len(data)} bytes", flush=True)


if __name__ == "__main__":
    main()
