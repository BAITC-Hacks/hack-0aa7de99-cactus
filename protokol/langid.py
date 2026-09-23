"""Определение языка участка речи — с вероятностями, а не одной меткой.

Используем энкодер Whisper напрямую (`detect_language`), без прогона
декодера: это на порядок дешевле полной расшифровки и даёт распределение
по языкам, на котором уже можно строить маршрутизацию.

Почему распределение, а не argmax. Ошибки несимметричны:
  - казахскую речь пустить в русскую модель — катастрофа (см. README);
  - русскую речь пустить в казахскую модель — потеря пунктуации, не более.
Поэтому порог смещён: достаточно скромной вероятности казахского,
чтобы отправить участок в казахскую модель.
"""
import functools
import numpy as np
import mlx.core as mx
import mlx_whisper
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, SAMPLE_RATE
from mlx_whisper.decoding import detect_language as _detect
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer

BASE = "mlx-community/whisper-large-v3-turbo"


@functools.lru_cache(maxsize=2)
def _model_and_tokenizer(repo: str = BASE):
    model = load_model(repo, dtype=mx.float16)
    tok = get_tokenizer(multilingual=model.is_multilingual,
                        num_languages=model.num_languages)
    return model, tok


def probs(audio: np.ndarray, repo: str = BASE) -> dict[str, float]:
    """audio — моно float32, 16 кГц. Возвращает {язык: вероятность}."""
    model, tok = _model_and_tokenizer(repo)
    n_mels = model.dims.n_mels
    mel = log_mel_spectrogram(audio.astype(np.float32), n_mels=n_mels)
    mel = pad_or_trim(mel, mlx_whisper.audio.N_FRAMES, axis=-2).astype(mx.float16)
    _, dists = _detect(model, mel[None], tok)
    return dists[0]


WINDOW = 4.0        # окно анализа внутри реплики, секунд
HOP = 2.0           # шаг окна


def route(audio: np.ndarray, kk_threshold: float = 0.45,
          sr: int = SAMPLE_RATE) -> tuple[str, dict]:
    """Решение, какой моделью распознавать реплику.

    Считаем язык не на реплике целиком, а по скользящим окнам, и берём
    МАКСИМУМ вероятности казахского. Причина: в смешанной речи русских
    слов больше, и на всей реплике казахский размывается — замер дал
    kk=0.095 на фразе «мен подрядчикпен сөйлесемін, до конца недели
    график беремін», после чего русская модель превратила её в «мен
    подрайшик пенсию лысымен». По окнам казахский участок виден отчётливо.

    Асимметрия ошибок оправдывает «чуть что — в казахскую модель»:
    она обучена на код-свитчинге и русские вставки передаёт верно,
    теряя только пунктуацию. Обратная ошибка уничтожает текст.
    """
    # Окно подстраиваем под длину реплики. На короткой реплике фиксированное
    # окно в 4 с накрывает её целиком, и вкрапление казахского тонет в
    # русском: замер дал kk=0.0 на фразе «Данияр Серикович, менде бір сұрақ
    # бар — а бюджет на это уже заложен?», где казахская вставка занимает
    # меньше половины. Дробим мельче, но не короче 1.5 с — ниже этого
    # определение языка недостоверно.
    dur = len(audio) / sr
    w = WINDOW if dur >= 2 * WINDOW else max(1.5, dur / 2.5)
    win, hop = int(w * sr), max(1, int(w * sr / 2))
    spans = ([audio[i:i + win] for i in range(0, max(1, len(audio) - win // 2), hop)]
             if len(audio) > win else [audio])

    best_kk, whole = 0.0, None
    for s in spans:
        if len(s) < 0.8 * sr:
            continue
        p = probs(s)
        whole = whole or p
        best_kk = max(best_kk, float(p.get("kk", 0.0)))

    p_all = probs(audio)
    p_kk_all, p_ru_all = float(p_all.get("kk", 0.0)), float(p_all.get("ru", 0.0))
    lang = "kk" if (best_kk >= kk_threshold or p_kk_all > p_ru_all) else "ru"
    top = dict(sorted(p_all.items(), key=lambda kv: -kv[1])[:3])
    info = {k: round(float(v), 3) for k, v in top.items()}
    info["kk_макс_по_окнам"] = round(best_kk, 3)
    return lang, info


MIN_SECONDS = 0.6


def route_span(samples: np.ndarray, sr: int = SAMPLE_RATE, **kw):
    """то же, но с защитой от слишком коротких реплик («Иә», «Понял»),
    на которых определение языка недостоверно"""
    if len(samples) < MIN_SECONDS * sr:
        return None, {}
    return route(samples, **kw)
