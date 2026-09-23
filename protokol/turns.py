"""Нарезка записи на реплики и распознавание каждой своей моделью.

Порядок обратный обычному: сначала диаризация, потом распознавание.

Так делается по двум причинам, обе подтверждены замером.

1. Границы. Если сначала распознать, а потом клеить с говорящими,
   границы сегментов Whisper (по паузам) не совпадают с границами
   pyannote (по смене голоса), и последнее слово предыдущего оратора
   уезжает в начало следующей реплики.

2. Вырождение. Whisper, получив кусок, обрезанный посреди фразы,
   досочиняет за край аудио и срывается в бессмыслицу. Реплика
   диаризации заканчивается на естественной паузе, поэтому обрыва нет.

Плюс язык определяется для каждой реплики отдельно: на совещании один
человек обычно говорит на одном языке, а переключение внутри фразы
берёт на себя казахская модель, обученная на код-свитчинге.
"""
import numpy as np
import asr, langid

SR = 16000
PAD = 0.20          # подушка по краям реплики, чтобы не срезать слово
MIN_TURN = 0.35     # реплики короче — мусор диаризации
MERGE_GAP = 0.60    # паузы короче склеиваем в одну реплику


def load_audio(path: str) -> np.ndarray:
    """моно float32 16 кГц через ffmpeg — без torchcodec и лишних зависимостей"""
    import subprocess
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
         "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def merge(turns: list[dict]) -> list[dict]:
    """склеиваем подряд идущие реплики одного говорящего"""
    out: list[dict] = []
    for t in sorted(turns, key=lambda x: x["start"]):
        if out and t["speaker"] == out[-1]["speaker"] and \
                t["start"] - out[-1]["end"] <= MERGE_GAP:
            out[-1]["end"] = max(out[-1]["end"], t["end"])
        else:
            out.append(dict(t))
    return [t for t in out if t["end"] - t["start"] >= MIN_TURN]


MULTILINGUAL_HINT = 0.30    # хватает одной уверенно казахской реплики


def is_multilingual(audio: np.ndarray, tt: list[dict]) -> tuple[bool, float]:
    """Есть ли в записи казахская речь вообще.

    Определение языка на энкодере дёшево, поэтому прогоняем его по всем
    репликам заранее. Если казахского нет нигде — вторую модель можно
    не грузить вовсе, и чисто русское совещание обрабатывается втрое
    быстрее.
    """
    best = 0.0
    for t in tt:
        a, b = int(t["start"] * SR), int(t["end"] * SR)
        if b - a < MIN_TURN * SR:
            continue
        _, info = langid.route(audio[a:b])
        best = max(best, info.get("kk_макс_по_окнам", 0.0),
                   info.get("kk", 0.0))
    return best >= MULTILINGUAL_HINT, best


def transcribe_turns(audio: np.ndarray, turns: list[dict],
                     prompt: str | None = None, roster: list[str] | None = None,
                     log=print) -> list[dict]:
    """Распознавание по репликам.

    Чисто русская запись — быстрый путь одной моделью.
    Есть казахский — каждая реплика декодируется обеими моделями, выбор
    делает арбитр по содержимому (см. asr.transcribe_arbitrated).
    Так поднимается и короткая вставка, на которой определение языка
    по звуку не срабатывает: «менде бір сұрақ бар» внутри русской фразы
    даёт всего kk=0.012, но в тексте видно сразу.
    """
    merged = merge(turns)
    multi, score = is_multilingual(audio, merged)
    log(f"    казахская речь: {'есть' if multi else 'не найдена'} "
        f"(максимум по репликам {score:.2f})"
        f"{' — вторая модель не используется' if not multi else ''}")

    roster_words = {w for n in (roster or []) for w in n.lower().split()}
    res = []
    for t in merged:
        a = max(0, int((t["start"] - PAD) * SR))
        b = min(len(audio), int((t["end"] + PAD) * SR))
        chunk = audio[a:b]
        if len(chunk) < MIN_TURN * SR:
            continue

        if multi:
            text, used, ratio = asr.transcribe_arbitrated(
                chunk, roster_words=roster_words, prompt=prompt)
            info = f"{used} (казахских слов {ratio:.0%})"
        else:
            text, used = asr.transcribe_ru(chunk, prompt), "ru"
            info = "ru"

        flag = " ⚠вырождение" if asr.is_degenerate(text) else ""
        log(f"  [{t['start']:6.1f}-{t['end']:6.1f}] {t['speaker']} "
            f"{info}{flag} | {text[:58]}")
        if text:
            res.append({**t, "lang": used, "text": text})
    return res


def as_dialogue(turns: list[dict], mapping: dict) -> str:
    """Стенограмма для извлечения поручений.

    Если говорящие опознаны — подписываем реплики, это помогает
    определить ответственного по обращению.

    Если нет — склеиваем сплошным текстом. Граница реплики диаризации
    не обязана совпадать с границей предложения: на тестовой записи
    фраза «тренингке смета дайындаңыз, к пятнице нужно» оказалась
    разрезана пополам, и модель собрала из половинок несуществующее
    поручение. Без подписей разрывать текст незачем.
    """
    named = {t["speaker"] for t in turns if mapping.get(t["speaker"])
             and "не опознан" not in mapping[t["speaker"]]}
    if len(named) < 2:
        return " ".join(t["text"].strip() for t in turns)

    lines, prev = [], None
    for t in turns:
        who = mapping.get(t["speaker"], t["speaker"])
        if who == prev:                      # продолжение той же реплики
            lines[-1] += " " + t["text"].strip()
        else:
            lines.append(f"{who}: {t['text'].strip()}")
            prev = who
    return "\n".join(lines)
