"""Двуязычный протокол: одно совещание — две версии документа.

Совещание идёт на смеси русского и казахского, а протокол в организации
нужен на одном языке — и в разных случаях на разном. Поэтому итоговый
документ собирается на любом из двух, из одних и тех же данных.

Что переводится: формулировки поручений, темы, саммари.
Что НЕ переводится ни при каких условиях: ФИО, названия организаций,
даты и номера. Имя и срок — это поля контроля исполнения, они обязаны
совпадать в обеих версиях побуквенно, иначе документ бесполезен.
Поэтому после перевода они восстанавливаются из оригинала программно,
а не «по просьбе» к модели.
"""
import copy
import llm

SYSTEM_KK = """Переведи текст с русского на казахский язык.

ЗАПРЕЩЕНО переводить, склонять или как-либо менять:
- имена и фамилии людей,
- названия организаций,
- даты, числа, номера договоров.
Их переноси в точности как в оригинале.

Верни СТРОГО JSON того же вида, что получил, с переведёнными значениями."""

SYSTEM_RU = SYSTEM_KK.replace("с русского на казахский", "с казахского на русский")

# поля, которые переводим
TRANSLATE = ("task", "area", "urgency")
# поля, которые обязаны остаться как есть
KEEP = ("assignee", "due_date", "due_raw", "due_source", "lang",
        "quote", "at_start", "at_end")


def translate(data: dict, to: str = "kk") -> dict:
    """Возвращает копию протокола на нужном языке."""
    src = copy.deepcopy(data)
    payload = {
        "topics": src.get("topics", []),
        "summary": src.get("summary", ""),
        "tasks": [{k: t.get(k) for k in TRANSLATE if t.get(k)}
                  for t in src.get("tasks", [])],
    }
    system = SYSTEM_KK if to == "kk" else SYSTEM_RU
    try:
        out = llm.generate_json(system, __import__("json").dumps(
            payload, ensure_ascii=False), max_tokens=2500, temp=0.0)
    except Exception:
        return src

    src["topics"] = out.get("topics") or src.get("topics", [])
    src["summary"] = out.get("summary") or src.get("summary", "")

    got = out.get("tasks") or []
    for i, t in enumerate(src.get("tasks", [])):
        if i >= len(got) or not isinstance(got[i], dict):
            continue
        for k in TRANSLATE:
            if got[i].get(k):
                t[k] = got[i][k]
        # поля контроля восстанавливаем из оригинала — на случай,
        # если модель всё же их тронула
        for k in KEEP:
            if k in data.get("tasks", [])[i]:
                t[k] = data["tasks"][i][k]
    src["lang_version"] = to
    return src
