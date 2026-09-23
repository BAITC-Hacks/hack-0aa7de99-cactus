# Local batch reproduction

This project runs meeting audio on Apple Silicon without a cloud inference API.
`batch_run.py` is a tracked project file. `smoke/` and `models/` are local, ignored
output directories; a fresh checkout does not need any files from either one.
Audio must already be on this Mac. The commands below use the two supplied
recordings in `~/Desktop/tracks/innovations/`; change the paths if needed.

## One-time setup

Use Python 3.12 on macOS with FFmpeg and `ffprobe` available on `PATH`:

```bash
cd ~/Downloads/track8
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p models/whisper models/diarization models/ollama-runtime smoke
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mlx-community/whisper-large-v3-turbo-q4',
                  revision='660c343bbf4e52ac257f0b7d952e5388e6f93bef',
                  local_dir='models/whisper')
snapshot_download(repo_id='pyannote-community/speaker-diarization-community-1',
                  revision='8a527374977391da736e0daaef26855d949d9685',
                  local_dir='models/diarization')
PY
curl -fL https://github.com/ollama/ollama/releases/download/v0.34.3/ollama-darwin.tgz -o /private/tmp/track8-ollama.tgz
tar -xzf /private/tmp/track8-ollama.tgz -C models/ollama-runtime
```

The model downloads contain weights only. Start the local Ollama service in its
own terminal, bound to loopback with cloud access disabled:

```bash
cd ~/Downloads/track8
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 \
OLLAMA_MODELS="$PWD/models/ollama" models/ollama-runtime/ollama serve
```

In another terminal, download the extraction model once:

```bash
cd ~/Downloads/track8
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$PWD/models/ollama" \
models/ollama-runtime/ollama pull qwen2.5:7b
```

## Run a complete recording

```bash
cd ~/Downloads/track8
export ASR_MODEL="$PWD/models/whisper"
export DIARIZATION_MODEL="$PWD/models/diarization"
export OLLAMA_MODEL=qwen2.5:7b
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -u batch_run.py \
  --audio "$HOME/Desktop/tracks/innovations/Совещание №2.mp3" \
  --output smoke/recording2
```

Substitute `Совещание №1.mp3` and `smoke/recording1` for recording #1.
Each output directory receives `asr.json`, `transcript.json`, `result.json`,
`protocol.docx` and `timings.json`. These contain meeting information and stay
local. If a run stops after a stage, use `--from-stage diarization`,
`--from-stage extraction`, or `--from-stage docx` with the same output path to
resume from saved files. `--stop-after` can end a run at an intermediate stage.
For example, after changing extraction code, regenerate both action inventories
without repeating speech models:

```bash
.venv/bin/python -u batch_run.py --output smoke/recording1 --from-stage extraction --stop-after extraction
.venv/bin/python -u batch_run.py --output smoke/recording2 --from-stage extraction --stop-after extraction
```

Optional `--review-note "..."` adds an explicit uncertainty for the reviewer;
it never changes transcript words. A reference protocol must be used for
comparison only, not fed to extraction. The local Qwen model proposes candidate
coverage; evidence checks decide which tasks are accepted. Unconfirmed items
remain in `review_candidates` in `result.json`.
