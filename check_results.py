"""Check local batch artifacts for missing words and unsupported action fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-actions", type=int)
    args = parser.parse_args()
    asr = read(args.output / "asr.json")
    source = read(args.output / "source_transcript.json")
    diar = read(args.output / "transcript.json")
    result = read(args.output / "result.json")
    assert len(asr) == len(source), "ASR source line count changed"
    assert [s["text"] for s in asr] == [s["text"] for s in source], "ASR source text changed"
    asr_words = [w["word"] for segment in asr for w in segment.get("words", [])]
    diar_words = [w["word"] for segment in diar for w in segment.get("words", [])]
    assert asr_words == diar_words, "Words were dropped or reordered during diarization"
    assert result["summary"].strip(), "Empty summary"
    actions = result["actions"]
    if args.expect_actions is not None:
        assert len(actions) == args.expect_actions, f"Expected {args.expect_actions} actions, got {len(actions)}"
    by_id = {line["id"]: line for line in source}
    seen = set()
    for index, action in enumerate(actions, 1):
        ids = action["evidence_ids"]
        assert ids and len(ids) == len(set(ids)), f"Action {index}: missing or repeated source IDs"
        assert all(i in by_id for i in ids), f"Action {index}: source ID outside transcript"
        assert tuple(ids) not in seen, f"Action {index}: duplicate evidence set"
        seen.add(tuple(ids))
        evidence = " | ".join(f'{by_id[i]["speaker"]} [{by_id[i]["time"]}]: {by_id[i]["text"]}' for i in ids)
        assert action["evidence"] == evidence, f"Action {index}: evidence differs from source"
        quoted = normalized(" ".join(by_id[i]["text"] for i in ids))
        for field in ("owner", "deadline"):
            value = normalized(action.get(field, ""))
            assert not value or value in quoted, f"Action {index}: {field} absent from cited text"
    print(f"{args.output}: {len(actions)} actions; {len(asr_words)} words preserved; "
          f"{sum(bool(x.get('needs_review')) for x in diar)} diarization spans for review")


if __name__ == "__main__":
    main()
