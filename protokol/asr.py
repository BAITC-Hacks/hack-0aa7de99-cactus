"""Распознавание речи с маршрутизацией по языку.

Две локальные модели, обе работают без обращения к внешним сервисам:

  ru → mlx-community/whisper-large-v3-turbo
       быстрая (MLX/Metal), даёт пунктуацию и заглавные

  kk → shyngys879/kazakh-whisper-large-v3-turbo
       LoRA-дообучение того же turbo на 841 тыс. примеров, включая KSC2
       с казахско-русским код-свитчингом. Держит имена и переключение
       языка внутри фразы, но выдаёт текст без пунктуации и в нижнем
       регистре, и работает через torch/MPS, то есть медленнее.

Замер на нашей тестовой записи (см. README): базовая модель превращает
«Ерлан Нұрланұлы» в «Ерлан Нұрланова», а «подрядчикпен» в «подрайшықпен».
Дообученная берёт оба слова верно. Поэтому маршрутизация, а не одна модель.
"""
import functools, re, warnings
import numpy as np

warnings.filterwarnings("ignore")

BASE = "mlx-community/whisper-large-v3-turbo"
KZ = "shyngys879/kazakh-whisper-large-v3-turbo"
SR = 16000


# ---------------------------------------------------------------- вырождение

def is_degenerate(text: str) -> bool:
    """Whisper на плохо знакомом языке срывается в повтор:
    «толық аудит жүргеніңіздіңіздіңізіңізіңіз…».
    Ловим по трём независимым признакам."""
    t = text.strip()
    if not t:
        return False
    words = t.split()

    # 1) один и тот же токен подряд много раз
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= 4:
        return True

    # 2) мало уникальных слов при большой длине
    if len(words) >= 12 and len(set(words)) / len(words) < 0.35:
        return True

    # 3) повтор слогов внутри одного длинного слова (ңіздіңіздіңіз)
    for w in words:
        if len(w) >= 14:
            for n in (3, 4, 5):
                chunk = w[-n:]
                if w.count(chunk) >= 3:
                    return True
    return False


# ------------------------------------------------------------------- модели

@functools.lru_cache(maxsize=1)
def _kz_pipe():
    import torch
    from transformers import pipeline
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=KZ,
                    device=dev, dtype=torch.float16)


def transcribe_ru(samples: np.ndarray, prompt: str | None = None) -> str:
    import mlx_whisper
    r = mlx_whisper.transcribe(
        samples.astype(np.float32), path_or_hf_repo=BASE, language="ru",
        initial_prompt=prompt, condition_on_previous_text=False,
        temperature=0.0, verbose=None)
    return r["text"].strip()


# Параметры против вырождения подобраны замером (см. README):
# без них казахская модель срывалась в цикл на 24-секундном участке.
KZ_GEN = {"language": "kk", "task": "transcribe",
          "no_repeat_ngram_size": 4, "repetition_penalty": 1.1}


def transcribe_kz(samples: np.ndarray) -> str:
    asr = _kz_pipe()
    out = asr({"raw": samples.astype(np.float32), "sampling_rate": SR},
              generate_kwargs=dict(KZ_GEN), return_timestamps=True)
    text = out["text"].strip()
    if is_degenerate(text):
        # вторая попытка: лучевой поиск устойчивее к срывам
        out = asr({"raw": samples.astype(np.float32), "sampling_rate": SR},
                  generate_kwargs=dict(KZ_GEN, num_beams=3),
                  return_timestamps=True)
        alt = out["text"].strip()
        if not is_degenerate(alt):
            return alt
        # обе попытки вырождены — отдаём короткую, она безопаснее
        return min(text, alt, key=len)
    return text


def transcribe(samples: np.ndarray, lang: str, prompt: str | None = None) -> str:
    return transcribe_kz(samples) if lang == "kk" else transcribe_ru(samples, prompt)


# ------------------------------------------------------- арбитраж двух моделей

KZ_LETTERS = set("әғқңөұүі")

# частотные казахские слова без специфических букв — иначе короткая
# вставка «менде бір сұрақ бар» опознаётся лишь наполовину
KZ_MARKERS = {
    "мен", "менде", "бар", "бір", "керек", "деп", "ғой", "және", "бұл", "осы",
    "сіз", "сізге", "бізге", "үшін", "болды", "емес", "жоқ", "қалай", "қандай",
    "бойынша", "туралы", "дейін", "кейін", "әлі", "тағы", "жақсы", "иә",
}


def kazakh_ratio(text: str, exclude: set[str] | None = None) -> float:
    """Доля казахских слов в тексте.

    Имена из справочника исключаются: «Сәуле» содержит ә, но её
    присутствие не означает, что реплика звучала по-казахски.
    """
    import re
    exclude = {w.lower() for w in (exclude or set())}
    words = [w for w in re.findall(r"[а-яёәғқңөұүі]+", text.lower())
             if w not in exclude]
    if not words:
        return 0.0
    hits = sum(1 for w in words
               if (set(w) & KZ_LETTERS) or w in KZ_MARKERS)
    return hits / len(words)


KZ_RATIO_THRESHOLD = 0.05


def transcribe_arbitrated(samples: np.ndarray, roster_words: set[str] | None = None,
                          prompt: str | None = None) -> tuple[str, str, float]:
    """Декодируем обеими моделями и выбираем по содержимому.

    Ни одна модель не лучше всегда, это показал замер:
      - на смешанной реплике русская выдала «мы дебр с Рахбар»
        вместо «менде бір сұрақ бар»;
      - на чисто русской казахская выдала «повязка» вместо «повестка»
        и потеряла пунктуацию.

    Решаем по выводу казахской модели: если в нём есть настоящие
    казахские слова (не считая имён из справочника) — берём её вариант,
    иначе русский, он точнее в русском и с пунктуацией.

    Возвращает (текст, какая модель, доля казахских слов).
    """
    ru = transcribe_ru(samples, prompt)
    kz = transcribe_kz(samples)
    ratio = kazakh_ratio(kz, roster_words)
    if is_degenerate(kz) and not is_degenerate(ru):
        return ru, "ru", ratio
    if ratio >= KZ_RATIO_THRESHOLD:
        return kz, "kk", ratio
    return ru, "ru", ratio
