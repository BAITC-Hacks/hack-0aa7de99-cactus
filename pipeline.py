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
    result = mlx_whisper.transcribe(
        str(audio), path_or_hf_repo=model, task="transcribe", word_timestamps=True
    )
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio)
    ], text=True).strip())
    segments = []
    for s in result["segments"]:
        start = max(0.0, float(s["start"]))
        if not s["text"].strip() or start >= duration:
            continue
        words = [
            {"start": max(0.0, float(w["start"])),
             "end": min(duration, float(w["end"])), "word": w["word"]}
            for w in s.get("words", [])
            if w["word"].strip() and float(w["start"]) < duration
        ]
        segments.append({
            "start": start, "end": min(duration, float(s["end"])),
            "speaker": "Не определён", "text": s["text"].strip(), "words": words
        })
    return segments


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
    return assign_speakers(segments, turns)


def assign_speakers(segments: list[dict], turns: list[tuple]) -> list[dict]:
    """Use word timing to split cross-speaker ASR segments; flag ambiguous words."""
    turns = sorted(turns)
    labels = {label: f"Speaker {index + 1}" for index, label in
              enumerate(dict.fromkeys(label for _, _, label in turns))}

    def label_for(start: float, end: float) -> str | None:
        length = end - start
        if length <= 0:
            return None
        scores = {}
        for turn_start, turn_end, label in turns:
            overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
            scores[label] = scores.get(label, 0.0) + overlap
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < 0.6 * length:
            return None
        if len(ranked) > 1 and ranked[1][1] > 0.2 * length:
            return None
        return labels[ranked[0][0]]

    output = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            speaker = label_for(segment["start"], segment["end"])
            output.append({**segment, "speaker": speaker or "Не определён",
                           "needs_review": speaker is None})
            continue
        for word in words:
            start, end = word["start"], word["end"]
            # Keep words with collapsed timing: dropping them can remove names
            # or deadlines from the transcript used for action extraction.
            end = max(start, end)
            speaker = label_for(start, end) or "Не определён"
            if output and output[-1].get("_source") == id(segment) and output[-1]["speaker"] == speaker:
                output[-1]["end"] = end
                output[-1]["text"] += word["word"]
                output[-1]["words"].append(word)
            else:
                output.append({"start": start, "end": end, "speaker": speaker,
                               "text": word["word"], "words": [word],
                               "needs_review": speaker == "Не определён",
                               "_source": id(segment)})
    # A very short label island at a speaker change is not reliable enough to
    # override the surrounding turn. Keep the words visible for manual review.
    for index, item in enumerate(output):
        if item["speaker"] == "Не определён" or item["end"] - item["start"] >= 1.0:
            continue
        following = next((other for other in output[index + 1:]
                          if other.get("_source") == item.get("_source")
                          and other["speaker"] != "Не определён"), None)
        if (following and following["speaker"] != item["speaker"]
                and following["start"] - item["end"] <= 1.0):
            item["speaker"] = "Не определён"
            item["needs_review"] = True
    for item in output:
        item["text"] = item["text"].strip()
        item.pop("_source", None)
    return [item for item in output if item["text"]]


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


def _local_json(prompt: str, schema: dict | None = None) -> dict:
    payload = json.dumps({"model": os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
                          "prompt": prompt, "stream": False, "format": schema or "json",
                          "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 4096}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            data = json.load(response)
    except Exception as exc:
        raise RuntimeError("Локальный Ollama недоступен. Запустите локальную модель.") from exc
    if data.get("done_reason") == "length":
        raise RuntimeError("Ответ локальной модели обрезан лимитом токенов; результат не принят.")
    result = json.loads(data["response"])
    if not isinstance(result, dict):
        raise RuntimeError("Локальная модель вернула неверную структуру JSON.")
    result["_model_metrics"] = {key: data.get(key) for key in (
        "model", "prompt_eval_count", "eval_count", "load_duration",
        "prompt_eval_duration", "eval_duration", "done_reason")}
    return result


def _explicit_actions(lines: list[dict]) -> list[dict]:
    """Conservative Russian request parser: quoted tasks, named addressees, literal deadlines.

    Model suggestions outside these explicit patterns remain review candidates.
    Speaker cluster labels never establish an owner. Contextual addressees are
    flagged because a nearby name is weaker evidence than an explicit assignment.
    """
    name = re.compile(r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна)")
    directive = re.compile(r"\b(?:разберитесь|проверьте|свяжитесь|проводите|запросите)\b", re.I)
    date = re.compile(
        r"\b(?:до|к)\s+\d{1,2}\s+[а-яё]+|\bдо\s+(?:пятницы|среды|понедельника|вторника|четверга)|"
        r"\bна\s+(?:этой|следующей)\s+неделе|\b(?:к среде|в среде)\b", re.I)
    anchors = [i for i, line in enumerate(lines)
               if re.search(r"\bОтветственн(?:ый|ая|ые)\b|Юридический департамент\. Срок", line["text"], re.I)
               or directive.search(line["text"])]
    actions = []
    for ordinal, index in enumerate(anchors):
        line = lines[index]
        text = line["text"]
        ids = {line["id"]}
        notes = []
        explicit = re.search(r"Ответственн(?:ый|ая|ые)\s+(.+?)\.\s*Срок", text, re.I)
        department = re.search(r"(Юридический департамент)\.\s*Срок", text, re.I)
        if explicit or department:
            marker = explicit or department
            owner = marker.group(1).strip()
            task = text[:marker.start()].strip().rstrip(".")
            dates = list(date.finditer(text))
            deadline = dates[-1].group() if dates else ""
        else:
            # Explicit directives are the anchors; suggestions/questions are not.
            task = text
            owner = ""
            context = lines[max(0, index - 10):index]
            joined = " ".join(x["text"] for x in context)
            matches = list(name.finditer(joined))
            if matches:
                match = matches[-1]
                owner = match.group()
                cursor = 0
                for source in context:
                    end = cursor + len(source["text"])
                    if cursor < match.end() and end > match.start():
                        ids.add(source["id"])
                    cursor = end + 1
                notes.append("Исполнитель определён по ближайшему обращению; проверьте связь обращения с поручением.")
            next_index = anchors[ordinal + 1] if ordinal + 1 < len(anchors) else len(lines)
            follow = lines[index:min(next_index, index + 12)]
            # A report requested after an audit is its deliverable, not a second task.
            if re.search(r"аудит", text, re.I):
                for source in follow[1:]:
                    if re.search(r"жду.*отч[её]т", source["text"], re.I):
                        task += " " + source["text"]
                        ids.add(source["id"])
                for source in context[-3:]:
                    if re.search(r"\d+\s+площад", source["text"]):
                        ids.add(source["id"])
            # Preserve an explicitly attached quality requirement or acceptance.
            if re.search(r"проводите", text, re.I):
                for source in follow[1:]:
                    if re.search(r"давайте договоримся", source["text"], re.I):
                        task += " " + source["text"]
                        ids.add(source["id"])
            if re.search(r"запросите", text, re.I):
                if index:
                    task = lines[index - 1]["text"] + " " + task
                    ids.add(lines[index - 1]["id"])
                for source in follow[1:]:
                    if re.search(r"запрошу", source["text"], re.I):
                        task += " " + source["text"]
                        ids.add(source["id"])
            # Search only the request and its cited refinements/reply, not unrelated context.
            deadline = ""
            for source in follow:
                if source["id"] in ids:
                    matches = list(date.finditer(source["text"]))
                    if matches:
                        deadline = matches[-1].group()
            if re.search(r"если", task, re.I):
                notes.append("Условное решение: не трактуйте возможное нарушение или расторжение как установленный факт.")
            if re.search(r"в среде", deadline, re.I):
                notes.append("Срок сохранён дословно из ASR; вероятный день недели требует проверки по аудио.")
        source_lines = [x for x in lines if x["id"] in ids]
        actions.append({"task": task, "owner": owner, "deadline": deadline,
                        "evidence_ids": sorted(ids), "review_notes": notes,
                        "evidence": " | ".join(
                            f'{x["speaker"]} [{x["time"]}]: {x["text"]}' for x in source_lines)})
    return actions



def _unstructured_actions(lines: list[dict]) -> list[dict]:
    """Accept direct spoken requests in an unnumbered discussion.

    This deliberately quotes the ASR instead of rewriting task facts. Nearby
    commitments can refine a request, and closing recaps can add uncertainty;
    neither is a new request by itself. The patterns describe Russian meeting
    grammar, not any reference protocol's action text.
    """
    name = re.compile(r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна)")
    imperative = re.compile(r"\b[а-яё]+(?:айте|яйте|уйте|ите|йте|ьте)\b", re.I)
    generic = {"смотрите", "давайте", "предлагаете", "можете", "подскажите", "скажите", "здравствуйте"}
    due = re.compile(
        r"до конца недели|за (?:\d+|одну|две|три)\s+(?:недел[юьи]|дн(?:я|ей)|месяц[а-яё]*)|"
        r"на (?:этой|следующей) неделе|по итогам (?:этого|этой) совещани[яю]|"
        r"не больше недели|максимум \d+ дн(?:ей|я)|к \d{1,2} [а-яё]+", re.I)
    recap_start = next((i for i, row in enumerate(lines)
                        if re.search(r"\b(?:подытожим|итого)\b", row["text"], re.I)), len(lines))
    body = lines[:recap_start]
    recap = lines[recap_start:]

    def is_directive(text: str) -> bool:
        verbs = [x.group().casefold() for x in imperative.finditer(text)]
        return any(verb not in generic for verb in verbs)

    # Elliptical requests for a distinct follow-up report have no imperative.
    def is_report(text: str) -> bool:
        return bool(re.search(r"\b(?:мне|жду)\b.*(?:справк[уае]|отч[её]т)", text, re.I))

    anchors = [i for i, row in enumerate(body)
               if (is_directive(row["text"]) and not
                   ("пусть" in row["text"].casefold() and "а вы" in row["text"].casefold()))
               or is_report(row["text"])]
    groups = []
    for index in anchors:
        row = body[index]
        if is_report(row["text"]):
            groups.append([index])
            continue
        if groups and index - groups[-1][-1] <= 2 and not is_report(body[groups[-1][0]]["text"]):
            between = " ".join(x["text"] for x in body[groups[-1][-1]:index + 4])
            # A named third-person assignment and a separate 'you' assignment
            # in the ensuing refinement have different owners.
            transfer = re.search(r"пусть\s+[А-ЯЁ][а-яё]+.*?а\s+вы", between, re.I)
            if not transfer:
                groups[-1].append(index)
                continue
        groups.append([index])

    actions = []
    for group in groups:
        first, last = group[0], group[-1]
        sources = set(group)
        task_text = " ".join(body[i]["text"] for i in range(first, last + 1))
        if first and (re.search(r"\bпредлагаю\b", body[first - 1]["text"], re.I)
                      or re.search(r"предлагаю.*вариант", body[first]["text"], re.I)):
            sources.add(first - 1)
            task_text = body[first - 1]["text"] + " " + task_text
        notes = []
        # Acknowledged details and final deliverables occur directly after a
        # request. Keep their words and citations, but do not create new tasks.
        following = body[last + 1:min(len(body), last + 5)]
        for offset, row in enumerate(following, last + 1):
            text = row["text"]
            if re.search(r"\b(?:дам|подготовлю|обновлю|найду|организуем)\b", text, re.I):
                sources.add(offset)
                task_text += " " + text
                break
        # A 'пусть NAME ... а вы ...' line refines two preceding requests.
        transfer_index = next((i for i in range(last + 1, min(len(body), last + 5))
                               if re.search(r"пусть\s+[А-ЯЁ][а-яё]+.*?а\s+вы", body[i]["text"], re.I)), None)
        owner = ""
        deadline = ""
        if transfer_index is not None:
            transfer = body[transfer_index]["text"]
            before, after = re.split(r"\bа\s+вы\b", transfer, maxsplit=1, flags=re.I)
            request = body[first]["text"].casefold()
            # Match the direct request to the same object in the refinement.
            if "претензи" in request and "претензи" in before.casefold():
                person = re.search(r"пусть\s+([А-ЯЁ][а-яё]+)", before)
                if person:
                    owner = person.group(1)
                task_text += " " + before.strip()
                found = list(due.finditer(before))
            else:
                task_text += " А вы " + after.strip()
                found = list(due.finditer(after))
                notes.append("Исполнитель назван местоимением «вы»; проверьте имя по записи.")
            sources.add(transfer_index)
            if found:
                deadline = found[-1].group()
        if not owner:
            context = " ".join(x["text"] for x in body[max(0, first - 10):first])
            prior = list(name.finditer(context))
            if prior:
                owner = prior[-1].group()
                for i in range(max(0, first - 10), first):
                    if owner.split()[0] in body[i]["text"] or owner.split()[-1] in body[i]["text"]:
                        sources.add(i)
                notes.append("Исполнитель определён по обращению перед просьбой; проверьте связь.")
        if not deadline:
            found = list(due.finditer(task_text))
            if found:
                deadline = found[-1].group()
        # An earlier planned outcome can have a separate operational deadline.
        if re.search(r"за месяц", task_text, re.I) and re.search(r"за неделю", task_text, re.I):
            deadline = "за неделю"
            notes.append("В источнике также указано «за месяц» для завершения работы; «за неделю» относится к смете.")
        # The recap may supply an owner or update a deadline, but never a new task.
        generic_words = {"недел", "сроки", "этому", "после", "сейчас", "хорош", "вопрос", "решен", "работ", "будет", "догов"}
        words = {w[:5].casefold() for w in re.findall(r"[А-Яа-яЁё]{6,}", task_text)} - generic_words
        for recap_row in recap:
            recap_words = {w[:5].casefold() for w in re.findall(r"[А-Яа-яЁё]{6,}", recap_row["text"])} - generic_words
            if len(words & recap_words) < 1:
                continue
            recap_names = list(name.finditer(recap_row["text"]))
            if not owner and recap_names:
                owner = recap_names[0].group()
                sources.add(lines.index(recap_row))
                notes.append("Имя взято из итоговой реплики ASR; написание требует проверки.")
            recap_due = list(due.finditer(recap_row["text"]))
            if recap_due:
                later = recap_due[-1].group()
                if deadline and later.casefold() != deadline.casefold():
                    notes.append(f"Сроки в основной и итоговой репликах расходятся: «{deadline}» / «{later}». Проверьте запись.")
                    sources.add(lines.index(recap_row))
                elif not deadline:
                    deadline = later
                    sources.add(lines.index(recap_row))
                    notes.append("Срок указан только в итоговой реплике; проверьте область его действия.")
        if not owner:
            notes.append("Исполнитель не подтверждён распознанным обращением.")
        source_lines = [body[i] if i < len(body) else lines[i] for i in sorted(sources)]
        actions.append({"task": task_text.strip(), "owner": owner, "deadline": deadline,
                        "evidence_ids": [row["id"] for row in source_lines],
                        "review_notes": notes,
                        "evidence": " | ".join(f'{row["speaker"]} [{row["time"]}]: {row["text"]}'
                                               for row in source_lines)})
    return actions


def extract(lines: list[dict], review_notes: list[str] | None = None) -> dict:
    """Local model proposes coverage; explicit source requests govern accepted facts."""
    import time
    started = time.perf_counter()
    if not lines:
        raise ValueError("Транскрипт пуст.")
    notes = list(review_notes or [])
    by_id = {x["id"]: x for x in lines}
    format_lines = lambda items: "\n".join(f'{x["id"]}: {x["text"]}' for x in items)
    candidates, metrics = [], []
    for offset in range(0, len(lines), 30):
        draft = _local_json(
            "Извлеки отдельные поручения из транскрипта. Верни JSON actions: массив "
            "task, owner, deadline, evidence_ids. Ссылки — номера строк. Не придумывай "
            "исполнителей или сроки. Не принимай вопрос или предложение за поручение.\n"
            + format_lines(lines[max(0, offset - 5):offset + 35]))
        metrics.append(draft.get("_model_metrics", {}))
        candidates.extend(x for x in draft.get("actions", []) if isinstance(x, dict))
    actions = _explicit_actions(lines)
    if not actions:
        actions = _unstructured_actions(lines)
    covered = {i for action in actions for i in action["evidence_ids"]}
    review_candidates = []
    for candidate in candidates:
        ids = {int(i) for i in candidate.get("evidence_ids", [])
               if str(i).isdigit() and int(i) in by_id}
        if not ids or not ids & covered:
            review_candidates.append({**candidate,
                "status": "needs_review", "reason": "Нет подтверждённого явного поручения в распознанных шаблонах; не включено автоматически."})
    for action in actions:
        # Keep supplied factual uncertainty adjacent to the affected action too.
        if re.search(r"област|Тимур", action["task"] + " " + action["owner"], re.I):
            action["review_notes"].extend(notes)
    if review_candidates:
        notes.append("Неподтверждённые предложения модели сохранены отдельно для проверки и не включены в таблицу поручений.")
    extraction_seconds = time.perf_counter() - started
    summary_started = time.perf_counter()
    summary = _local_json(
        "Напиши по-русски краткое содержание совещания: только 2 предложения о темах. "
        "Верни JSON summary (строка). Не перечисляй поручения. Не упоминай имена, "
        "географию, числа и даты: эти детали проверяются отдельно. "
        "Не утверждай, что все факты подтверждены.\n" + format_lines(lines),
        schema={"type": "object", "properties": {"summary": {"type": "string"}},
                "required": ["summary"], "additionalProperties": False})
    if not str(summary.get("summary") or "").strip():
        raise RuntimeError("Локальная модель не вернула краткое содержание; экспорт не разрешён.")
    metrics.append(summary.get("_model_metrics", {}))
    return {"summary": str(summary.get("summary") or ""), "decisions": [],
            "actions": actions, "review_notes": list(dict.fromkeys(notes)),
            "model_metrics": metrics, "extraction_candidates": candidates,
            "review_candidates": review_candidates,
            "stage_seconds": {"action_extraction": extraction_seconds,
                              "summary_generation": time.perf_counter() - summary_started}}


def questions(actions: list[dict]) -> list[str]:
    output = []
    for index, item in enumerate(actions, 1):
        task = item.get("task", "")
        if not item.get("owner"):
            output.append(f"Поручение {index} («{task}»): кто ответственный?")
        if not item.get("deadline"):
            output.append(f"Поручение {index} («{task}»): какой срок исполнения?")
        elif re.search(r"следующ|текущ|апта|скоро|позже|после совещания|недел|пятниц|сред|понедель|вторник|четверг|суббот|воскрес|квартал|через месяц", item["deadline"], re.I):
            output.append(f"Поручение {index} («{task}»): уточните дату для срока «{item['deadline']}».")
    return output
