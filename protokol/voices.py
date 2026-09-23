"""Голосовой справочник: опознание участника по тембру.

Зачем. Основной способ узнать, кто говорит, — разобрать обращения
(«Гульмира Сериковна, вам слово»). Он точен, но работает только если
к человеку хоть раз обратились по имени. Председателя, молчуна или
того, кого зовут «коллега», так не опознать.

Поэтому pyannote community-1 отдаёт вместе с разметкой ещё и
`speaker_embeddings` — вектор голоса на каждого найденного говорящего.
Сохранив их один раз, на следующем совещании человека узнаём по голосу,
без всяких обращений.

Справочник лежит в обычном JSON рядом с проектом. Никуда не передаётся:
это биометрия, и по условиям кейса она обязана оставаться в контуре.
"""
import json, os
import numpy as np

STORE = "voiceprints.json"
THRESHOLD = 0.62        # косинусная близость, ниже — считаем «не он»


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def load(path: str = STORE) -> dict[str, list]:
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def save(store: dict, path: str = STORE) -> None:
    json.dump(store, open(path, "w"), ensure_ascii=False)


def enroll(store: dict, name: str, embedding) -> dict:
    """Добавить образец голоса. Несколько образцов на человека
    усредняются — голос меняется от записи к записи (микрофон,
    простуда, громкость)."""
    vec = np.asarray(embedding, dtype=float).tolist()
    store.setdefault(name, []).append(vec)
    store[name] = store[name][-5:]      # держим последние пять
    return store


def identify(store: dict, embedding, threshold: float = THRESHOLD):
    """Кто это? Возвращает (имя, близость) или (None, лучшая близость)."""
    if not store or embedding is None:
        return None, 0.0
    v = np.asarray(embedding, dtype=float)
    best, score = None, 0.0
    for name, samples in store.items():
        s = max(_cos(v, np.asarray(x, dtype=float)) for x in samples)
        if s > score:
            best, score = name, s
    return (best, score) if score >= threshold else (None, score)


def identify_all(store: dict, embeddings: dict[str, object],
                 threshold: float = THRESHOLD) -> dict[str, str]:
    """SPEAKER_XX → ФИО для всех, кого удалось узнать.

    Одно имя не может достаться двум говорящим: разбираем пары
    по убыванию близости.
    """
    pairs = []
    for spk, emb in embeddings.items():
        if emb is None:
            continue
        v = np.asarray(emb, dtype=float)
        for name, samples in store.items():
            s = max(_cos(v, np.asarray(x, dtype=float)) for x in samples)
            if s >= threshold:
                pairs.append((s, spk, name))
    pairs.sort(reverse=True)
    out, used = {}, set()
    for s, spk, name in pairs:
        if spk in out or name in used:
            continue
        out[spk] = name
        used.add(name)
    return out
