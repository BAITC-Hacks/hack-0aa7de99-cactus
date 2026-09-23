#!/usr/bin/env python3
"""Автопротоколирование совещаний. Всё локально, без облачных API.

    аудио
      → диаризация (pyannote community-1)
      → определение языка по репликам (энкодер Whisper, по окнам)
      → распознавание: ru → MLX turbo, kk/смешанное → казахское дообучение
      → восстановление пунктуации в казахских репликах (локальная LLM)
      → коррекция имён по справочнику организации
      → привязка говорящих к ФИО по обращениям
      → извлечение поручений (локальная LLM)
      → протокол DOCX

Запуск:
    python run.py запись.m4a --roster roster.json --date 2026-09-22
"""
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")

import diarize, turns, glossary, normalize, extract, protocol, asr, voices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--roster", help="JSON-список участников (из приглашения)")
    ap.add_argument("--date", default="2026-09-22")
    ap.add_argument("--weekday", default="вторник")
    ap.add_argument("--out", default="Протокол.docx")
    ap.add_argument("--no-punct", action="store_true",
                    help="не восстанавливать пунктуацию в казахских репликах")
    a = ap.parse_args()

    roster = json.load(open(a.roster)) if a.roster else []
    T0 = time.time()
    step = lambda n, s: print(f"\n[{n}] {s}", flush=True)

    # --- 1. кто когда говорил
    step(1, "Различение говорящих…")
    t = time.time()
    n_spk = len(roster) or None
    cache = f".diar_{os.path.basename(a.audio)}_{n_spk}.json"
    raw_turns, voiceprints = diarize.diarize_cached(a.audio, cache, num_speakers=n_spk)
    n_found = len({x["speaker"] for x in raw_turns})
    print(f"    {time.time()-t:.0f}с, говорящих: {n_found}"
          f"{'' if not n_spk else f' (ожидалось {n_spk})'}")

    # --- 2. язык + распознавание по репликам
    step(2, "Распознавание с маршрутизацией по языку…")
    t = time.time()
    audio = turns.load_audio(a.audio)
    prompt = ("Совещание. Участники: " + ", ".join(roster) + ".") if roster else None
    tt = turns.transcribe_turns(audio, raw_turns, prompt=prompt, roster=roster)
    n_kk = sum(1 for x in tt if x["lang"] == "kk")
    n_bad = sum(1 for x in tt if asr.is_degenerate(x["text"]))
    print(f"    {time.time()-t:.0f}с, реплик: {len(tt)} "
          f"(казахских/смешанных: {n_kk}), вырожденных: {n_bad}")

    # --- 3. пунктуация в казахских репликах
    if not a.no_punct and n_kk:
        step(3, "Восстановление пунктуации в казахских репликах…")
        t = time.time()
        import llm
        kept = 0
        for x in tt:
            if x["lang"] != "kk":
                continue
            new, fid = normalize.restore(x["text"], llm.generate)
            if new != x["text"]:
                kept += 1
            else:
                print(f"    реплика оставлена как есть (точность {fid:.2f})")
            x["text"] = new
        print(f"    {time.time()-t:.0f}с, обработано {kept} из {n_kk}")

    # --- 4. имена собственные
    step(4, "Коррекция имён по справочнику…")
    if roster:
        for x in tt:
            x["text"], fx = glossary.correct(x["text"], roster)
            for o, n in fx:
                print(f"    {o} → {n}")

    # --- 5. кто есть кто
    step(5, "Привязка говорящих к ФИО…")
    mapping = diarize.label_speakers(tt, roster) if roster else {}
    for sp, nm in sorted(mapping.items()):
        print(f"    по обращению:  {sp} → {nm}")

    # Голосовой справочник дополняет опознание по обращениям: к человеку
    # могли ни разу не обратиться по имени (председатель, молчун), зато
    # его голос знаком по прошлым совещаниям.
    store = voices.load()
    if voiceprints:
        by_voice = voices.identify_all(store, voiceprints)
        for sp, nm in by_voice.items():
            if sp not in mapping:
                mapping[sp] = nm
                print(f"    по голосу:     {sp} → {nm}")
            elif mapping[sp] != nm:
                print(f"    ⚠ расхождение: {sp} — обращение «{mapping[sp]}»,"
                      f" голос «{nm}»; оставлено обращение")
        # пополняем справочник теми, кого опознали надёжно
        added = 0
        for sp, nm in mapping.items():
            if sp in voiceprints and "не опознан" not in nm:
                voices.enroll(store, nm, voiceprints[sp]); added += 1
        if added:
            voices.save(store)
            print(f"    в справочник голосов добавлено образцов: {added}"
                  f" (всего людей: {len(store)})")

    dialogue = turns.as_dialogue(tt, mapping)

    # --- 6. поручения
    step(6, "Извлечение поручений…")
    t = time.time()
    data = extract.run(dialogue, meeting_date=a.date, weekday=a.weekday)
    # Справочник применяем ещё раз — уже к результату. LLM переписывает
    # имена своими словами и роняет казахскую графику: в замере она
    # выдала «Ерлан Нурланұлы» вместо «Нұрланұлы», хотя в стенограмме
    # было верно. Поле «ответственный» должно совпадать со справочником.
    if roster:
        for x in data.get("tasks", []):
            if x.get("assignee"):
                x["assignee"] = glossary.correct(x["assignee"], roster)[0]
        for p in data.get("participants", []):
            if p.get("name"):
                p["name"] = glossary.correct(p["name"], roster)[0]
    print(f"    {time.time()-t:.0f}с, поручений: {len(data.get('tasks', []))}")

    # --- 7. документ
    step(7, "Сборка протокола…")
    protocol.build(data, out=a.out, date=a.date, transcript=dialogue)
    json.dump({"transcript": dialogue, "turns": tt, "speakers": mapping, **data},
              open(a.out.replace(".docx", ".json"), "w"),
              ensure_ascii=False, indent=1)

    print(f"\nГотово за {time.time()-T0:.0f}с → {a.out}\n")
    print(f"{'ОТВЕТСТВЕННЫЙ':<22} {'СРОК':<12} {'ЯЗ':<6} ПОРУЧЕНИЕ")
    for x in data.get("tasks", []):
        due = str(x.get("due_date") or x.get("due_raw") or "—")[:11]
        print(f"{str(x.get('assignee','—'))[:21]:<22} {due:<12} "
              f"{str(x.get('lang','—')):<6} {x.get('task','')[:52]}")


if __name__ == "__main__":
    main()
