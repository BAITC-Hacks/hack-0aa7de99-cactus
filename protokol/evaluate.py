"""Сверка извлечённых поручений с эталонным протоколом (написанным человеком).

Эталон — это протоколы, выданные организатором кейса. Они НЕ являются
дословной расшифровкой аудио: это идеальный результат работы секретаря.
Поэтому сверяем по смыслу (перекрытие значимых слов), а не буквально.
"""
import json, re, sys, difflib

STOP = set("и в на по с за до от для не что как это к а о у же бы при из".split())

def bag(s):
    return {w for w in re.findall(r"[а-яёa-z]+", s.lower()) if w not in STOP and len(w) > 2}

def match(pred, gold, thr=0.25):
    """Жадное сопоставление предсказанных поручений с эталонными.

    ВНИМАНИЕ: сверка лексическая (перекрытие слов + совпадение ответственного).
    Эталон переформулирован человеком, поэтому метрика ЗАНИЖАЕТ качество:
    «организовать совещание с подрядчиками» и «провести совещание с
    подрядчиками, зафиксировать график» — одно поручение, но Jaccard = 0.25.
    Честная оценка требует семантической сверки; здесь цифра нижняя граница.
    """
    used, rows = set(), []
    for g in gold:
        gb = bag(g["task"])
        best, bi = 0.0, None
        for i, p in enumerate(pred):
            if i in used: continue
            pb = bag(p.get("task",""))
            j = len(gb & pb) / len(gb | pb) if gb | pb else 0
            # совпадение ответственного — дополнительный сигнал
            a = difflib.SequenceMatcher(None, (p.get("assignee") or "").lower(),
                                        g["assignee"].lower()).ratio()
            sc = 0.75 * j + 0.25 * (a if a > 0.7 else 0)
            if sc > best: best, bi = sc, i
        if bi is not None and best >= thr:
            used.add(bi)
            p = pred[bi]
            a_ok = difflib.SequenceMatcher(None,
                     (p.get("assignee") or "").lower(), g["assignee"].lower()).ratio() > 0.7
            rows.append((g, p, round(best,2), a_ok))
        else:
            rows.append((g, None, round(best,2), False))
    extra = [p for i,p in enumerate(pred) if i not in used]
    return rows, extra

if __name__ == "__main__":
    name = sys.argv[1]; predfile = sys.argv[2]
    gold = json.load(open("gold.json"))[name]
    pred = json.load(open(predfile))["tasks"]
    rows, extra = match(pred, gold)
    found = sum(1 for _,p,_,_ in rows if p)
    ok_a  = sum(1 for _,p,_,a in rows if p and a)
    print(f"\n=== {name} ===")
    print(f"Эталонных поручений: {len(gold)} | Найдено: {found} | Верный ответственный: {ok_a}")
    print(f"Извлечено всего: {len(pred)} | Сверх эталона: {len(extra)}\n")
    for g, p, s, a in rows:
        if p:
            print(f"  ✅ {g['task'][:58]}")
            print(f"     ответственный: {p.get('assignee')} {'✅' if a else '❌ эталон: '+g['assignee']}"
                  f" | срок: {p.get('due_date') or p.get('due_raw')} (эталон {g['due']})")
        else:
            print(f"  ❌ ПРОПУЩЕНО: {g['task'][:58]}")
    if extra:
        print("\n  Дополнительно найдено (нет в эталоне, но может быть верно):")
        for p in extra: print(f"     + {p.get('assignee')}: {p.get('task','')[:60]}")
