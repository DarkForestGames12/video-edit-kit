#!/usr/bin/env bash
# video-edit-kit setup: Python deps, speech model, matting model, fonts, icons.
# Everything heavy is cached in $VEK_HOME (default ~/.vek). Safe to re-run.
set -euo pipefail
VEK_HOME="${VEK_HOME:-$HOME/.vek}"
mkdir -p "$VEK_HOME/models" "$VEK_HOME/assets"
echo "[vek] python packages"
pip install --break-system-packages -q sherpa-onnx onnxruntime kaldi-native-fbank soundfile scipy numpy opencv-contrib-python-headless pillow pyyaml 2>&1 | grep -v -i warning || true
command -v ffmpeg >/dev/null || { echo "[vek] ffmpeg missing: apt-get install -y ffmpeg"; exit 1; }
cd "$VEK_HOME/models"
if [ ! -f sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx ]; then
  echo "[vek] speech model (Parakeet TDT 0.6B, ~480MB, from GitHub releases)"
  curl -sSL --retry 3 -o parakeet.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
  tar xjf parakeet.tar.bz2 && rm parakeet.tar.bz2
fi
if [ ! -f sherpa-onnx-nemo-ctc-en-conformer-small/model.int8.onnx ]; then
  echo "[vek] alignment model (NeMo Conformer-CTC small, ~76MB)"
  curl -sSL --retry 3 -o ctc.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
  tar xjf ctc.tar.bz2 && rm ctc.tar.bz2
fi
if [ ! -f rvm_mobilenetv3_fp32.onnx ]; then
  echo "[vek] matting model (Robust Video Matting)"
  curl -sSL --retry 3 -o rvm_mobilenetv3_fp32.onnx https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
fi
cd "$VEK_HOME/assets"
for p in @expo-google-fonts/montserrat @expo-google-fonts/anton @expo-google-fonts/space-grotesk @expo-google-fonts/bebas-neue @expo-google-fonts/poppins @expo-google-fonts/inter @fortawesome/fontawesome-free; do
  d=$(echo "$p" | sed 's#[@/]#_#g')
  if [ ! -d "$d" ]; then
    echo "[vek] npm asset $p"
    t=$(npm pack --silent "$p" 2>/dev/null | tail -1) && mkdir -p "$d" && tar xzf "$t" -C "$d" && rm -f "$t"
  fi
done
echo "[vek] ready -> $VEK_HOME"
