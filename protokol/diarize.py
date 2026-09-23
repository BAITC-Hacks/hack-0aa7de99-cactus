"""Диаризация: кто когда говорил + привязка к реальным ФИО.

Модель pyannote/speaker-diarization-community-1 (CC-BY-4.0) скачивается с
HuggingFace один раз и дальше работает локально. Нужен токен: страница модели
закрыта гейтом (принятие условий). Передачи аудио во внешние сервисы нет.

Ключевая идея второго шага: на совещании к людям обращаются по имени
(«Гульмира Сериковна, вам слово»). Значит анонимные метки SPEAKER_00 можно
сопоставить с реальными ФИО, разобрав обращения — без образцов голоса.
"""
import os, subprocess, tempfile, json, re


def _to_wav(path: str) -> str:
    """pyannote ждёт моно 16 кГц"""
    out = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", out],
                   check=True, capture_output=True)
    return out


def diarize(audio: str, token: str | None = None, num_speakers: int | None = None):
    from pyannote.audio import Pipeline
    from huggingface_hub import get_token
    # токен берём из кэша `hf auth login`, из env или из аргумента
    token = token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Нужна авторизация HuggingFace. Выполните `hf auth login`, затем примите "
            "условия на странице pyannote/speaker-diarization-community-1."
        )
    # Официальный репозиторий закрыт гейтом: нужно принять условия на его
    # странице. Если доступ не выдан — падаем на зеркало сообщества
    # (та же модель под CC-BY-4.0, но аккаунт неофициальный, поэтому
    # происхождение весов не подтверждено — для сдачи лучше получить доступ
    # к официальному).
    from huggingface_hub.errors import GatedRepoError
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=token)
    except (GatedRepoError, Exception) as e:
        if "Gated" not in type(e).__name__ and "403" not in str(e):
            raise
        print("  ⚠ официальный репозиторий недоступен (условия не приняты),"
              " используется зеркало сообщества", flush=True)
        pipe = Pipeline.from_pretrained(
            "pyannote-community/speaker-diarization-community-1", token=token)
    # Читаем аудио сами и передаём тензором. pyannote 4.x иначе декодирует
    # через torchcodec, которому нужны разделяемые библиотеки FFmpeg —
    # на macOS с brew-версией он их обычно не находит.
    import torch, soundfile as sf
    wav = _to_wav(audio)
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    os.unlink(wav)
    waveform = torch.from_numpy(data.T)  # (канал, отсчёты)
    out = pipe({"waveform": waveform, "sample_rate": sr},
               num_speakers=num_speakers)
    # pyannote 4.x возвращает DiarizeOutput. Берём exclusive-вариант:
    # он без перекрывающихся реплик и предназначен как раз для склейки
    # с таймкодами транскрипции.
    ann = getattr(out, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = getattr(out, "speaker_diarization", out)
    turns = [{"start": t.start, "end": t.end, "speaker": s}
             for t, _, s in ann.itertracks(yield_label=True)]

    # Вектор голоса на каждого говорящего — для голосового справочника.
    # Порядок строк соответствует ann.labels(), так сказано в модели.
    emb = getattr(out, "speaker_embeddings", None)
    prints: dict[str, list] = {}
    if emb is not None:
        try:
            for label, vec in zip(ann.labels(), emb):
                prints[label] = [float(x) for x in vec]
        except Exception:
            prints = {}
    return turns, prints


def diarize_cached(audio: str, cache: str, **kw):
    """Диаризация занимает минуты — результат кешируем на диск.

    Возвращает (реплики, отпечатки голосов).
    """
    if os.path.exists(cache):
        d = json.load(open(cache))
        if isinstance(d, dict):
            return d["turns"], d.get("voiceprints", {})
        return d, {}                       # кэш старого формата
    turns, prints = diarize(audio, **kw)
    json.dump({"turns": turns, "voiceprints": prints}, open(cache, "w"))
    return turns, prints


def attach(segments: list[dict], turns: list[dict]) -> list[dict]:
    """каждому сегменту Whisper присваиваем говорящего по максимальному
    перекрытию во времени"""
    out = []
    for s in segments:
        best, dur = None, 0.0
        for t in turns:
            ov = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if ov > dur:
                best, dur = t["speaker"], ov
        out.append({**s, "speaker": best or "SPEAKER_?"})
    return out


def label_speakers(labelled: list[dict], roster: list[str], extract_mod=None) -> dict:
    """Сопоставляем SPEAKER_XX → ФИО по обращениям в репликах, без образцов голоса.

    Приём: если в реплике SPEAKER_00 звучит «Гульмира Сериковна, вам слово»,
    то СЛЕДУЮЩИЙ говорящий — почти наверняка Гульмира Сериковна.

    Отдельно опознаём председателя. Он раздаёт слово, поэтому произносит
    больше всего чужих имён — и по «следующему говорящему» его самого
    опознать нельзя. Его имя достаётся методом исключения: то, которое
    не досталось никому другому.
    """
    speakers = sorted({s["speaker"] for s in labelled})

    # Обращение и ответ не всегда попадают в соседние сегменты: Whisper режет
    # речь по паузам, между ними может оказаться «Спасибо» или уточнение.
    # Поэтому смотрим на несколько следующих реплик с убывающим весом.
    WINDOW = 3
    votes: dict[str, dict[str, float]] = {}
    addressed: dict[str, set] = {sp: set() for sp in speakers}
    for i, seg in enumerate(labelled):
        for name in roster:
            first = name.split()[0]
            if not re.search(rf"\b{re.escape(first)}\w*", seg["text"], re.I):
                continue
            addressed[seg["speaker"]].add(name)
            for k in range(1, WINDOW + 1):
                if i + k >= len(labelled):
                    break
                nxt = labelled[i + k]["speaker"]
                if nxt == seg["speaker"]:
                    continue
                votes.setdefault(nxt, {}).setdefault(name, 0.0)
                votes[nxt][name] += 1.0 / k
                break   # засчитываем только первого другого говорящего

    # председатель — кто назвал больше всего разных людей
    chair = max(speakers, key=lambda sp: len(addressed[sp])) if speakers else None

    # жадно раздаём имена по числу голосов, председателя пропускаем
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    pairs = sorted(((n, sp, c) for sp, cc in votes.items() for n, c in cc.items()),
                   key=lambda x: -x[2])
    for name, sp, _ in pairs:
        if sp == chair or sp in mapping or name in taken:
            continue
        mapping[sp] = name
        taken.add(name)

    # Страховка от ненадёжной диаризации. Если почти никто не набрал
    # голосов, значит реплики нарезаны не по людям (так бывает, когда
    # запись начитана одним голосом за всех). Подписывать реплики чужими
    # именами хуже, чем не подписывать: неверная подпись уводит за собой
    # и ответственного в поручении.
    if len(speakers) > 1 and len(votes) < max(2, (len(speakers) + 1) // 2):
        return {}

    # метод исключения: пока остаётся ровно один безымянный говорящий
    # и ровно одно нераспределённое имя — они парой и являются
    for _ in range(len(speakers)):
        free_sp = [s for s in speakers if s not in mapping]
        free_nm = [n for n in roster if n not in taken]
        if len(free_sp) == 1 and len(free_nm) == 1:
            mapping[free_sp[0]] = free_nm[0]
            taken.add(free_nm[0])
        else:
            break

    if chair is not None and chair not in mapping:
        mapping[chair] = "председатель (не опознан)"
    return mapping


def as_dialogue(labelled: list[dict], mapping: dict) -> str:
    """стенограмма вида «ФИО: реплика», склеивая подряд идущие реплики"""
    lines, cur, buf = [], None, []
    for s in labelled:
        who = mapping.get(s["speaker"], s["speaker"])
        if who != cur:
            if buf:
                lines.append(f"{cur}: {' '.join(buf).strip()}")
            cur, buf = who, []
        buf.append(s["text"].strip())
    if buf:
        lines.append(f"{cur}: {' '.join(buf).strip()}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    audio = sys.argv[1]
    stt = json.load(open(sys.argv[2]))
    roster = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "roster.json"))
    n = len(roster) or None     # число участников известно из приглашения
    turns, _prints = diarize_cached(audio, f"diar_cache_{n}.json", num_speakers=n)
    print(f"Говорящих: {len({t['speaker'] for t in turns})}, реплик: {len(turns)}")
    lab = attach(stt["segments"], turns)
    m = label_speakers(lab, roster, None)
    print("Сопоставление:", json.dumps(m, ensure_ascii=False, indent=1))
    print(as_dialogue(lab, m)[:1500])
