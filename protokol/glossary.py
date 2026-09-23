"""Коррекция имён собственных в стенограмме по справочнику организации.

Обе модели распознавания коверкают редкие имена, каждая по-своему:
  русская   — «Болатович» → «Булатович», «Сәуле Маратовна» → «Савли Маратовна»
  казахская — реже, но выдаёт всё в нижнем регистре

Для протокола это критично: искажается поле «ответственный».

Два приёма, оба нужны:

1. Фонетическая нормализация. Русская модель пишет казахские звуки
   русскими буквами: ә→а, ұ→у, қ→к. Сравнивать надо после приведения
   к общему виду, иначе «сәуле» и «сауле» выглядят разными словами.

2. Сопоставление по ПОЛНОМУ имени, а не по отдельным словам.
   «Савли» против «Сәуле» — сходство 0.4, не поймать. Но «Савли
   Маратовна» против «Сәуле Маратовна» — 0.79, потому что отчество
   уцелело. Опознав пару, чиним только испорченное слово.

Уже правильные падежные формы не трогаем: «Нұрланұлыға» остаётся как
есть. Стенограмма должна быть дословной — контроль в __main__.
"""
import re, difflib

RU, KZ = "а-яё", "әғқңөұүhі"
ALPHA, UPPER = RU + KZ, "А-ЯЁӘҒҚҢӨҰҮІ"
VOWELS = "аеёиоуыэюяәөұүі"
WORD = rf"[{ALPHA}{UPPER}][{ALPHA}]{{2,}}"

_PHON = str.maketrans("әөұүіқғңһ", "аоууикгнх")

ENDINGS = ("", "а", "у", "ом", "ем", "е", "ы", "и", "ой", "ей", "ю", "я",
           "ах", "ами", "ның", "нің", "ға", "ге", "ды", "ді", "мен", "бен",
           "ұлы", "ича", "ичем", "ичу", "вна", "вны", "вне")


def phon(s: str) -> str:
    return s.lower().translate(_PHON)


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, phon(a), phon(b), autojunk=False).ratio()


def _stem(term: str) -> str:
    return term[:-1] if term and term[-1].lower() in VOWELS else term


def _variants(term: str):
    st = _stem(term)
    seen = set()
    for e in ENDINGS:
        v = st + e
        if v and v not in seen:
            seen.add(v)
            yield v


def orthographic_fix(tok: str, term: str) -> str | None:
    """Токен звучит как термин — но правильно ли записан?

    Разделяем два разных случая, которые легко перепутать:

      «Сауле»        — основа записана по-русски вместо казахской (Сәул…);
                       чиним основу, окончание оставляем → «Сәуле»
      «Нұрланұлыға»  — основа верная, дальше идёт падежное окончание;
                       не трогаем вовсе

    Поэтому сравниваем не слово целиком, а только его начало длиной
    с каноническую основу.

    Возвращает исправленный токен, либо None если он уже верен.
    """
    st = _stem(term)
    if len(tok) < len(st):
        return None
    head = tok[:len(st)]
    if phon(head) != phon(st):
        return None                      # это вообще не тот термин
    if head == st or head.lower() == st.lower():
        return None                      # написание верное, падеж не наш случай
    return st + tok[len(st):]


def _is_correct_form(tok: str, term: str) -> bool:
    """форма допустима: либо канонична, либо отличается только падежом"""
    st = _stem(term)
    if len(tok) >= len(st) and phon(tok[:len(st)]) == phon(st):
        return orthographic_fix(tok, term) is None
    return any(sim(tok, v) > 0.97 for v in _variants(term))


def correct(text: str, glossary: list[str],
            phrase_threshold: float = 0.74, token_threshold: float = 0.84):
    """Возвращает (исправленный текст, список замен)."""
    toks = list(re.finditer(WORD, text))
    words = [m.group(0) for m in toks]
    repl: dict[int, str] = {}
    fixes = []

    def put(idx: int, new: str):
        old = words[idx]
        if new.lower() == old.lower():
            return
        if old[0].isupper() or idx == 0:
            new = new[0].upper() + new[1:]
        repl[idx] = new
        fixes.append((old, new, None))

    # --- 1. многословные имена: опознаём пару, чиним испорченное слово
    for term in sorted(glossary, key=lambda t: -len(t.split())):
        parts = term.split()
        n = len(parts)
        if n < 2:
            continue
        for i in range(len(words) - n + 1):
            if any(j in repl for j in range(i, i + n)):
                continue
            cand = " ".join(words[i:i + n])
            s = sim(cand, term)
            if s < phrase_threshold:
                continue
            # хотя бы одно слово пары должно быть опознано уверенно —
            # иначе это случайное совпадение длин
            if not any(_is_correct_form(words[i + k], parts[k]) or
                       sim(words[i + k], parts[k]) >= 0.8 for k in range(n)):
                continue
            for k, part in enumerate(parts):
                w = words[i + k]
                ortho = orthographic_fix(w, part)
                if ortho is not None:            # верный звук, неверные буквы
                    put(i + k, ortho)
                elif not _is_correct_form(w, part):
                    put(i + k, part)

    # --- 2. одиночные слова справочника
    singles = [t for term in glossary for t in term.split() if len(t) >= 5]
    for i, w in enumerate(words):
        if i in repl or len(w) < 5:
            continue
        best, score, ending = None, 0.0, ""
        for term in singles:
            if _is_correct_form(w, term):
                best = None
                break
            for v in _variants(term):
                s = sim(w, v)
                if s > score:
                    best, score, ending = term, s, v[len(_stem(term)):]
        if best and token_threshold <= score < 0.995:
            put(i, _stem(best) + ending)

    if not repl:
        return text, []
    out, last = [], 0
    for i, m in enumerate(toks):
        if i in repl:
            out.append(text[last:m.start()]); out.append(repl[i]); last = m.end()
    out.append(text[last:])
    return "".join(out), [(a, b) for a, b, _ in fixes]


if __name__ == "__main__":
    import json, sys
    G = json.load(open(sys.argv[1]))
    text = sys.stdin.read()
    fixed, fixes = correct(text, G)
    print(f"Замен: {len(fixes)}")
    for a, b in fixes:
        print(f"  {a!r:<24} → {b!r}")
    print("\n" + fixed)
