"""Локальная LLM (Qwen3-14B, MLX). Один загруженный экземпляр на процесс.

Обращений к внешним сервисам нет: веса лежат в локальном кэше, вывод
считается на Metal. Это требование ТЗ, а не предпочтение.
"""
import functools, json, re

# 14B вместо 8B по результатам замера на тестовой записи:
# 8B нашла 3 поручения из 5, 14B — 5 из 5 с верными ответственными.
# Ценой вдвое большего времени (126 с против 52 с) и 8.3 ГБ весов.
MODEL = "mlx-community/Qwen3-14B-4bit"


@functools.lru_cache(maxsize=1)
def _load():
    from mlx_lm import load
    return load(MODEL)


def generate(system: str, user: str, max_tokens: int = 2500,
             temp: float = 0.1) -> str:
    from mlx_lm import generate as _gen
    from mlx_lm.sample_utils import make_sampler
    model, tok = _load()
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    return _gen(model, tok, prompt=prompt, max_tokens=max_tokens,
                sampler=make_sampler(temp=temp), verbose=False)


def generate_json(system: str, user: str, **kw) -> dict:
    """то же, но с извлечением JSON из ответа и понятной ошибкой"""
    raw = generate(system, user, **kw)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"модель не вернула JSON: {raw[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # частый случай — оборванный по лимиту токенов хвост
        s = m.group(0)
        for cut in range(len(s) - 1, 0, -1):
            if s[cut] == "}":
                try:
                    return json.loads(s[:cut + 1] + "]}"[: s[:cut].count("[") -
                                                         s[:cut].count("]")])
                except Exception:
                    continue
        raise
