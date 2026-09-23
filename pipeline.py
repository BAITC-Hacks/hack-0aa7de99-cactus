"""Local-only meeting pipeline. Models must already exist on the host."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path


def transcribe(audio: Path) -> list[dict]:
    model = os.environ.get("ASR_MODEL", "")
    if not model or not Path(model).exists():
        raise RuntimeError("Укажите ASR_MODEL: локальный путь к MLX Whisper модели.")
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError("Установите mlx-whisper на Mac с Apple Silicon.") from exc
    # language=None enables automatic language detection; no forced translation.
    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=model, task="transcribe")
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio)
    ], text=True).strip())
    return [
        {"start": max(0.0, float(s["start"])),
         "end": min(duration, float(s["end"])),
         "speaker": "Не определён", "text": s["text"].strip()}
        for s in result["segments"]
        if s["text"].strip() and float(s["start"]) < duration
    ]


def diarize(audio: Path, segments: list[dict]) -> list[dict]:
    model = os.environ.get("DIARIZATION_MODEL", "")
    if not model or not Path(model).exists():
        raise RuntimeError("Укажите DIARIZATION_MODEL: локальный pipeline pyannote.")
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("Установите pyannote.audio для диаризации.") from exc
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio),
                        "-ac", "1", "-ar", "16000", wav.name], check=True)
        import torch
        import wave
        with wave.open(wav.name, "rb") as source:
            sample_rate = source.getframerate()
            samples = source.readframes(source.getnframes())
        waveform = torch.frombuffer(bytearray(samples), dtype=torch.int16).float()
        waveform = (waveform / 32768.0).unsqueeze(0)
        pipeline = Pipeline.from_pretrained(model)
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        annotation = getattr(output, "exclusive_speaker_diarization", output)
        turns = [(turn.start, turn.end, label)
                 for turn, _, label in annotation.itertracks(yield_label=True)]
    labels = {}
    for s in segments:
        scores = {}
        for start, end, label in turns:
            overlap = max(0.0, min(s["end"], end) - max(s["start"], start))
            scores[label] = scores.get(label, 0.0) + overlap
        if scores and max(scores.values()) > 0:
            label = max(scores, key=scores.get)
            if label not in labels:
                labels[label] = f"Speaker {len(labels) + 1}"
            s["speaker"] = labels[label]
    return segments


def stamp(seconds: float) -> str:
    return f"{int(seconds // 60):02}:{int(seconds % 60):02}"


def lines_from_segments(segments: list[dict]) -> list[dict]:
    return [dict(id=i + 1, time=stamp(s["start"]), **s)
            for i, s in enumerate(segments)]


def parse_manual(text: str) -> list[dict]:
    """For demo/debug only: one `Name: utterance` per line."""
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = re.match(r"^([^:]{1,100}):\s*(.+)$", raw.strip())
        if not m:
            raise ValueError(f"Нужен формат «Имя: текст»: {raw[:100]}")
        lines.append(dict(id=len(lines) + 1, time="—", start=0.0,
                          end=0.0, speaker=m[1].strip(), text=m[2].strip()))
    return lines


def extract(lines: list[dict]) -> dict:
    """Ask local Ollama for structured facts; validate every citation afterwards."""
    if not lines:
        raise ValueError("Транскрипт пуст.")
    transcript = "\n".join(f'{x["id"]} | {x["time"]} | {x["speaker"]}: {x["text"]}'
                           for x in lines)
    prompt = (
        "Ты секретарь совещания. Исходный текст может быть русским, казахским или смешанным. "
        "Верни ТОЛЬКО JSON object с ключами summary (строка), decisions (массив строк), "
        "actions (массив объектов: task, owner, deadline, evidence_ids — массив номеров строк). "
        "Включай только действительно произнесённые поручения. Ответственный — назначенный "
        "исполнитель, не обязательно говорящий. Не придумывай имена и даты. Если нет срока или "
        "исполнителя, поставь пустую строку. Сохраняй относительный срок дословно. "
        "Для каждого поручения укажи строку с прямым доказательством; если поручение уточняется "
        "позже, добавь и строку уточнения. Не дублируй поручения при итоговом повторении.\n\n"
        + transcript
    )
    payload = json.dumps({"model": os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
                          "prompt": prompt, "stream": False, "format": "json"}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            data = json.load(response)
    except Exception as exc:
        raise RuntimeError("Локальный Ollama недоступен. Запустите ollama serve и загрузите модель.") from exc
    result = json.loads(data["response"])
    by_id = {x["id"]: x for x in lines}
    actions = []
    for item in result.get("actions", []):
        ids = [int(i) for i in item.get("evidence_ids", []) if str(i).isdigit()]
        ids = list(dict.fromkeys(i for i in ids if i in by_id))
        if not ids or not str(item.get("task", "")).strip():
            continue  # Never present unsupported action items as verified.
        actions.append({"task": str(item["task"]).strip(),
                        "owner": str(item.get("owner") or "").strip(),
                        "deadline": str(item.get("deadline") or "").strip(),
                        "evidence_ids": ids,
                        "evidence": " | ".join(f'{by_id[i]["speaker"]} [{by_id[i]["time"]}]: '
                                               f'{by_id[i]["text"]}' for i in ids)})
    return {"summary": str(result.get("summary") or ""),
            "decisions": [str(x) for x in result.get("decisions", [])],
            "actions": actions}


def questions(actions: list[dict]) -> list[str]:
    output = []
    for index, item in enumerate(actions, 1):
        task = item.get("task", "")
        if not item.get("owner"):
            output.append(f"Поручение {index} («{task}»): кто ответственный?")
        if not item.get("deadline"):
            output.append(f"Поручение {index} («{task}»): какой срок исполнения?")
        elif re.search(r"следующ|текущ|апта|аптасы|скоро|позже|после совещания", item["deadline"], re.I):
            output.append(f"Поручение {index} («{task}»): уточните дату для срока «{item['deadline']}».")
    return output
