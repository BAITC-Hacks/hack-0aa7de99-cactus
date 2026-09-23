#!/usr/bin/env python3
"""Веб-интерфейс: загрузка записи → протокол с поручениями.

Сервер поднимается на localhost и никуда не ходит — весь расчёт идёт
теми же модулями, что и CLI. Обработка одной записи занимает минуты,
поэтому она выполняется в фоновом потоке, а страница опрашивает статус
и показывает, на каком из семи шагов находится работа.

Запуск:
    python web.py           → http://127.0.0.1:5050
"""
import json, os, threading, time, traceback, uuid, warnings
from pathlib import Path
from datetime import date as today_date
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_file, Response

import diarize, turns, glossary, normalize, extract, protocol, asr, voices, extracts
import anchors, gaps, bilingual

app = Flask(__name__)
JOBS: dict[str, dict] = {}
BASE = Path(__file__).resolve().parent
UPLOADS = str(BASE / "uploads")
RESULTS = str(BASE / "results")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

STEPS = ["Различение говорящих", "Распознавание речи", "Пунктуация",
         "Имена по справочнику", "Кто есть кто", "Поручения",
         "Незакрытые решения", "Протокол"]


def _set(job: dict, step: int, note: str = "") -> None:
    job["step"] = step
    job["note"] = note
    job["log"].append({"step": step, "name": STEPS[step - 1], "note": note,
                       "at": round(time.time() - job["t0"], 1)})


def process(job_id: str, audio: str, roster: list, date: str, weekday: str) -> None:
    job = JOBS[job_id]
    try:
        n_spk = len(roster) or None
        _set(job, 1)
        cache = str(BASE / f".diar_{os.path.basename(audio)}_{n_spk}.json")
        raw_turns, prints = diarize.diarize_cached(audio, cache, num_speakers=n_spk)
        found = len({x["speaker"] for x in raw_turns})
        _set(job, 1, f"{found} говорящих")

        _set(job, 2)
        wave = turns.load_audio(audio)
        prompt = ("Совещание. Участники: " + ", ".join(roster) + ".") if roster else None
        tt = turns.transcribe_turns(wave, raw_turns, prompt=prompt, roster=roster,
                                    log=lambda *_: None)
        n_kk = sum(1 for x in tt if x["lang"] == "kk")
        _set(job, 2, f"{len(tt)} реплик, казахских {n_kk}")

        _set(job, 3)
        if n_kk:
            import llm
            for x in tt:
                if x["lang"] == "kk":
                    x["text"], _ = normalize.restore(x["text"], llm.generate)
        _set(job, 3, "готово" if n_kk else "не требуется")

        _set(job, 4)
        fixes = 0
        if roster:
            for x in tt:
                x["text"], fx = glossary.correct(x["text"], roster)
                fixes += len(fx)
        _set(job, 4, f"{fixes} правок")

        _set(job, 5)
        mapping = diarize.label_speakers(tt, roster) if roster else {}
        how = {sp: "по обращению" for sp in mapping}
        store = voices.load()
        if prints:
            for sp, nm in voices.identify_all(store, prints).items():
                if sp not in mapping:
                    mapping[sp], how[sp] = nm, "по голосу"
            for sp, nm in mapping.items():
                if sp in prints and "не опознан" not in nm:
                    voices.enroll(store, nm, prints[sp])
            voices.save(store)
        dialogue = turns.as_dialogue(tt, mapping)
        _set(job, 5, f"опознано {len([v for v in mapping.values() if 'не опознан' not in v])}")

        _set(job, 6)
        data = extract.run(dialogue, meeting_date=date, weekday=weekday)
        if roster:
            for x in data.get("tasks", []):
                if x.get("assignee"):
                    x["assignee"] = glossary.correct(x["assignee"], roster)[0]
        _set(job, 6, f"{len(data.get('tasks', []))} поручений")

        # таймкоды: цитата-основание становится ссылкой на секунду записи
        anchors.attach(data.get("tasks", []), tt)

        _set(job, 7)
        holes = gaps.find(dialogue, data.get("tasks"))
        anchors.attach(holes, tt)
        _set(job, 7, f"{len(holes)} пробелов")

        _set(job, 8)
        docx = os.path.join(RESULTS, f"{job_id}.docx")
        protocol.build(data, out=docx, date=date, transcript=dialogue)
        extracts.build(data, out_dir=os.path.join(RESULTS, job_id + "_выписки"),
                       date=date)
        _set(job, 8, "готово")

        job["data"] = data
        job["turns"] = tt
        job["date"] = date
        job["result"] = {
            "tasks": data.get("tasks", []),
            "gaps": holes,
            "audio": f"/audio/{job_id}",
            "summary": data.get("summary", ""),
            "topics": data.get("topics", []),
            "transcript": dialogue,
            "speakers": [{"name": n, "how": how.get(s, "по обращению")}
                         for s, n in sorted(mapping.items())],
            "elapsed": round(time.time() - job["t0"]),
            "docx": f"/download/{job_id}",
        }
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["trace"] = traceback.format_exc()


@app.post("/api/run")
def api_run():
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify({"error": "Выберите аудиофайл"}), 400
    job_id = uuid.uuid4().hex[:12]
    path = os.path.join(UPLOADS, job_id + "_" + Path(f.filename).name)
    f.save(path)
    try:
        roster = json.loads(request.form.get("roster") or "[]")
        roster = [s.strip() for s in roster if s.strip()]
    except json.JSONDecodeError:
        roster = []
    date = request.form.get("date") or today_date.today().isoformat()
    weekday = request.form.get("weekday") or ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"][today_date.fromisoformat(date).weekday()]
    JOBS[job_id] = {"state": "running", "step": 0, "note": "", "log": [],
                    "t0": time.time(), "steps": STEPS}
    threading.Thread(target=process, args=(job_id, path, roster, date, weekday),
                     daemon=True).start()
    return jsonify({"job": job_id})


@app.get("/api/status/<job_id>")
def api_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "нет такой задачи"}), 404
    return jsonify({k: v for k, v in job.items() if k not in ("t0", "trace")})


@app.get("/audio/<job_id>")
def audio(job_id):
    """отдаём загруженную запись, чтобы плеер мог перейти к моменту поручения"""
    for f in os.listdir(UPLOADS):
        if f.startswith(job_id + "_"):
            return send_file(os.path.join(UPLOADS, f))
    return "не найдено", 404


@app.post("/api/translate/<job_id>")
def api_translate(job_id):
    job = JOBS.get(job_id)
    if not job or "data" not in job:
        return jsonify({"error": "нет такой задачи"}), 404
    to = (request.json or {}).get("to", "kk")
    key = "tr_" + to
    if key not in job:
        job[key] = bilingual.translate(job["data"], to=to)
    d = job[key]
    return jsonify({"tasks": d.get("tasks", []), "summary": d.get("summary", ""),
                    "topics": d.get("topics", [])})


@app.post("/api/save/<job_id>")
def api_save(job_id):
    """правки человека перед экспортом: пересобираем протокол и выписки"""
    job = JOBS.get(job_id)
    if not job or "data" not in job:
        return jsonify({"error": "нет такой задачи"}), 404
    edited = (request.json or {}).get("tasks")
    if not isinstance(edited, list):
        return jsonify({"error": "ожидался список поручений"}), 400
    job["data"]["tasks"] = edited
    job["result"]["tasks"] = edited
    docx = os.path.join(RESULTS, f"{job_id}.docx")
    protocol.build(job["data"], out=docx, date=job.get("date", ""),
                   transcript=job["result"].get("transcript", ""))
    extracts.build(job["data"], out_dir=os.path.join(RESULTS, job_id + "_выписки"),
                   date=job.get("date", ""))
    return jsonify({"ok": True, "tasks": len(edited)})


@app.get("/download/<job_id>")
def download(job_id):
    path = os.path.join(RESULTS, f"{job_id}.docx")
    if not os.path.exists(path):
        return "не найдено", 404
    return send_file(path, as_attachment=True, download_name="Протокол.docx")


@app.get("/")
def index():
    return Response((BASE / "ui.html").read_text(encoding="utf-8"), mimetype="text/html")


if __name__ == "__main__":
    print("Интерфейс: http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, threaded=True)
