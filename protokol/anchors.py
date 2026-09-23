"""Привязка поручения к моменту записи.

У каждого поручения уже есть цитата-основание. Здесь она превращается
из текста в ссылку на секунду записи: находим реплику, из которой цитата
взята, и берём её таймкод.

Работает поверх готового результата — ни распознавание, ни извлечение
поручений не затрагиваются.
"""
import re, difflib


def _words(s: str) -> list[str]:
    return re.findall(r"[а-яёәғқңөұүіa-z0-9]+", (s or "").lower())


def _overlap(a: list[str], b: list[str]) -> float:
    """доля слов цитаты, найденных в реплике"""
    if not a:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / len(a)


def locate(quote: str, turns: list[dict], threshold: float = 0.45):
    """Возвращает (start, end) реплики, откуда цитата, либо None.

    Цитата может охватывать несколько реплик подряд, поэтому пробуем
    и одиночные, и пары соседних.
    """
    q = _words(quote)
    if len(q) < 3:
        return None
    best, score = None, threshold
    for i, t in enumerate(turns):
        s = _overlap(q, _words(t.get("text")))
        if s > score:
            best, score = (t["start"], t["end"]), s
        if i + 1 < len(turns):
            pair = _words(t.get("text")) + _words(turns[i + 1].get("text"))
            s2 = _overlap(q, pair)
            if s2 > score:
                best, score = (t["start"], turns[i + 1]["end"]), s2
    return best


def attach(tasks: list[dict], turns: list[dict]) -> list[dict]:
    """проставляет каждому поручению at_start / at_end в секундах"""
    for t in tasks:
        pos = locate(t.get("quote") or t.get("task") or "", turns)
        if pos:
            t["at_start"], t["at_end"] = round(pos[0], 1), round(pos[1], 1)
    return tasks
