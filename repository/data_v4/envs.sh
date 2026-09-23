#!/usr/bin/env bash
# Generator environments on the VESSL workspace (image quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim).
# Base Python 3.13 is too new for most TTS/music repos -> one uv venv per backend with its own Python.
# cu128 torch wheels run on the image's CUDA 13 driver. Usage: bash data_v4/envs.sh xtts chatterbox ...
set -euo pipefail
ENVS=${ENVS:-/root/envs}; MODELS=${MODELS:-/root/gen_models}; mkdir -p "$ENVS" "$MODELS"
command -v uv >/dev/null || pip install -q uv
TORCH="torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128"
mk() { uv venv -q -p "$2" "$ENVS/$1"; }
pipi() { uv pip install -q -p "$ENVS/$1/bin/python" "${@:2}"; }

for b in "$@"; do case "$b" in
  xtts)        mk xtts 3.11;       pipi xtts $TORCH; pipi xtts "coqui-tts[codec]" pyarrow soundfile ;;
  chatterbox)  mk chatterbox 3.11; pipi chatterbox $TORCH; pipi chatterbox chatterbox-tts pyarrow soundfile ;;
  melo)        mk melo 3.10;       pipi melo $TORCH; pipi melo "git+https://github.com/myshell-ai/MeloTTS.git" pyarrow soundfile
               "$ENVS/melo/bin/python" -m unidic download ;;
  cosyvoice2)  mk cosyvoice2 3.10; git -C "$MODELS" clone -q --recursive https://github.com/FunAudioLLM/CosyVoice.git || true
               pipi cosyvoice2 $TORCH; pipi cosyvoice2 -r "$MODELS/CosyVoice/requirements.txt" pyarrow
               "$ENVS/cosyvoice2/bin/hf" download FunAudioLLM/CosyVoice2-0.5B --local-dir "$MODELS/CosyVoice2-0.5B"
               echo "export PYTHONPATH=$MODELS/CosyVoice:$MODELS/CosyVoice/third_party/Matcha-TTS" > "$ENVS/cosyvoice2/pythonpath" ;;
  seedvc)      mk seedvc 3.10;     git -C "$MODELS" clone -q https://github.com/Plachtaa/seed-vc.git || true
               pipi seedvc $TORCH; pipi seedvc -r "$MODELS/seed-vc/requirements.txt" pyarrow ;;
  acestep)     mk acestep 3.10;    pipi acestep $TORCH; pipi acestep "git+https://github.com/ace-step/ACE-Step.git" ;;
  data)        mk data 3.11;       pipi data $TORCH; pipi data demucs==4.0.1 librosa==0.10.2.post1 soundfile pyarrow \
                                     scikit-learn huggingface_hub modelscope ;;
  *) echo "unknown backend $b" >&2; exit 2 ;;
esac; echo "env ready: $b"; done
