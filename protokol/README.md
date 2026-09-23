# Local meeting protocol pipeline

This application turns a local recording into a DOCX protocol with a transcript, speaker names, summary, and actions with owners and deadlines. It supports Russian, Kazakh, and mixed speech. The models run locally on Apple Silicon; processing does not use a cloud inference API.

## Setup

Install Python 3.12 and FFmpeg. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r protokol/requirements.txt
```

The diarization model is gated. Sign in to Hugging Face, accept the terms for [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1), then run `.venv/bin/hf auth login` locally. Never commit the token.

Download these model weights to the local Hugging Face cache before running offline:

```bash
.venv/bin/hf download mlx-community/whisper-large-v3-turbo
.venv/bin/hf download shyngys879/kazakh-whisper-large-v3-turbo
.venv/bin/hf download pyannote/speaker-diarization-community-1
.venv/bin/hf download mlx-community/Qwen3-14B-4bit
```

The downloads total roughly 12 GB. Keep the cache and any copies of weights outside Git. Model access and compatibility may depend on the installed library versions.

## Demo

```bash
.venv/bin/python protokol/web.py
```

Open <http://127.0.0.1:5050>. Select a local audio file, set its date, and optionally enter participant names. Uploaded audio, transcripts, generated DOCX files, and voiceprints stay in ignored local directories. The web server binds only to `127.0.0.1`.

For the command line:

```bash
.venv/bin/python protokol/run.py /path/to/local-recording.m4a \
  --date 2026-09-23 --weekday среда --out /path/to/local-output.docx
```

You may supply `--roster /path/to/local-roster.json`, a JSON array of participant names. Store it outside Git. Run `.venv/bin/python protokol/offline_check.py ...` with the same arguments after all models are cached to verify that processing makes no external socket connection.

The original source commit includes example meeting results and rosters. They are deliberately excluded from this submission; no recordings or transcripts are needed to review the code or launch the demo.
