#!/usr/bin/env bash

echo "====================================="
echo "  Installing MeloTTS Dependencies"
echo "  using uv (safe + isolated)"
echo "====================================="

# 1. 프로젝트 루트 이동 (원하면 수정)
cd ~/myrepos/MeloTTS || exit 1

echo "[1] Pin Python to 3.11"
uv python pin 3.11

echo "[2] Sync base environment"
uv sync

echo "[3] Installing core TTS dependencies..."
uv add txtsplit torch torchaudio cached_path

echo "[4] Installing NLP / tokenizers..."
uv add transformers==4.27.4 num2words==0.5.12 langid==1.1.6

echo "[5] Installing Japanese text processing..."
uv add unidic_lite==1.0.8 unidic==1.1.0 mecab-python3==1.0.9 \
       pykakasi==2.2.1 fugashi==1.3.0

echo "[6] Installing English / IPA tooling..."
uv add g2p_en==2.1.0 anyascii==0.3.2 eng_to_ipa==0.0.2 \
       inflect==7.0.0 unidecode==1.3.7

echo "[7] Install Korean G2P..."
uv add jamo==0.4.1 g2pkk>=0.1.1

echo "[8] Chinese text normalization..."
uv add pypinyin==0.50.0 jieba==0.42.1 cn2an==0.5.22

echo "[9] European multilingual text..."
uv add "gruut[de,es,fr]==2.2.3"

echo "[10] Audio processing..."
uv add librosa==0.9.1 pydub==0.25.1 soundfile tensorboard==2.16.2

echo "[11] Logging & UI tools..."
uv add loguru==0.7.2 tqdm gradio#!/usr/bin/env bash
set -e

echo "====================================================="
echo "  Installing ALL MeloTTS Dependencies (uv version)"
echo "====================================================="

# MeloTTS repo 경로로 이동 (원하면 수정)
cd ~/myrepos/MeloTTS || exit 1

echo "[1] Pin Python version to 3.11"
uv python pin 3.11

echo "[2] Sync base environment"
uv sync

echo "[3] Install PyTorch + torchaudio"
uv add "torch==2.3.1" "torchaudio==2.3.1"

echo "[4] Audio libraries"
uv add librosa==0.9.1 pydub==0.25.1 soundfile tensorboard==2.16.2

echo "[5] Progress bar / logging"
uv add tqdm loguru

echo "[6] Transformers + NLP"
uv add transformers==4.27.4 num2words==0.5.12 langid==1.1.6 cached_path

echo "[7] Japanese text processing (MeloTTS 내부 필요)"
uv add unidic_lite==1.0.8 unidic==1.1.0 mecab-python3==1.0.9 \
       pykakasi==2.2.1 fugashi==1.3.0

echo "[8] English phoneme tools"
uv add g2p_en==2.1.0 anyascii==0.3.2 eng_to_ipa==0.0.2 \
       inflect==7.0.0 unidecode==1.3.7

echo "[9] Korean text (g2p) tools"
uv add jamo==0.4.1 "g2pkk>=0.1.1" python-mecab-ko

echo "[10] Chinese text normalization"
uv add pypinyin==0.50.0 jieba==0.42.1 cn2an==0.5.22

echo "[11] European multilingual text (gruut)"
uv add "gruut[de,es,fr]==2.2.3"

echo "[12] Web UI (Gradio)"
uv add gradio

echo "-----------------------------------------------------"
echo " Downloading unidic dictionary (Japanese MeCab)"
echo "-----------------------------------------------------"
uv run python -m unidic download

echo "-----------------------------------------------------"
echo " Testing Korean g2p (MeCab-KO)"
echo "-----------------------------------------------------"
uv run python - << 'EOF'
from g2pkk import G2p
print("[TEST] g2pkk →", G2p()("안녕하세요 테스트입니다."))
EOF

echo "-----------------------------------------------------"
echo " All dependencies installed successfully!"
echo "-----------------------------------------------------"
echo "Test with:"
echo "   uv run python -c \"from melo.api import TTS; print('MeloTTS OK')\""
echo "-----------------------------------------------------"


echo "====================================="
echo "All dependencies installed successfully!"
echo "Test with:"
echo "   uv run python -c \"from melo.api import TTS; print('MeloTTS OK')\""
echo "====================================="
