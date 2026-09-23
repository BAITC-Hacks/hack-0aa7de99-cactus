# HackAlem Cactus — local meeting protocol

This private submission contains two local implementations. The newer Russian/Kazakh meeting protocol solution is in [`protokol/`](protokol/README.md). The original Track 8 implementation remains at the repository root; its instructions are in [`LOCAL_RUN.md`](LOCAL_RUN.md).

The `protokol` pipeline performs speaker diarization, Russian/Kazakh speech recognition, action extraction, and DOCX generation on the user's machine. Its web demo starts with:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r protokol/requirements.txt
.venv/bin/python protokol/web.py
```

Open <http://127.0.0.1:5050>. Download the models first as described in [`protokol/README.md`](protokol/README.md). Model weights, recordings, transcripts, generated protocols, participant rosters, and voiceprints belong on the local machine and are excluded from this repository.

The `protokol` code is adapted from [`adiletexe/hackalem-protokol`](https://github.com/adiletexe/hackalem-protokol), commit `409ecee9322349e7b37148b5420c5fb43b383ddb`, by `adiletexe`. Its source commit contains generated meeting material, so this repository imports the application code only. See [`SOURCE_ATTRIBUTION.md`](SOURCE_ATTRIBUTION.md).
