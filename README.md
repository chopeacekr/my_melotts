# 🎤 MeloTTS
## 📦 설치

```bash
git clone https://github.com/yourname/MeloTTS.git
cd MeloTTS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 빠른 시작

```python
from melotts import MeloTTS

tts = MeloTTS("KR")
audio = tts.speak("안녕하세요. 멜로 TTS 한국어 모델입니다.", speaker_id=0)
tts.save_wav(audio, "sample.wav")
```

## 1. 데이터 특성

- 다국어·멀티스피커 음성 데이터 기반
- 스튜디오 품질 중심의 고음질 음성
- 중국어 모델은 Chinese + English code-mixing 지원
- 한국어, 일본어, 스페인어, 프랑스어 등 다언어 포함

## 2. 학습 방식 / 모델 구조

- VITS / Bert-VITS2 기반 비자동회귀(non-autoregressive) 구조
- 텍스트 인코더: BERT 계열 사용
- 언어 ID + speaker ID 기반 멀티언어·멀티스피커 학습
- HiFi-GAN 계열 Vocoder 적용
- Alignment는 모델 내부에서 자동 학습(self-aligned)

## 3. 사용 라이브러리

- 핵심: torch, torchaudio
- G2P:
  - 영어: g2p_en, eng_to_ipa
  - 중국어: pypinyin, jieba
  - 한국어: g2pkk, jamo
  - 일본어: mecab, unidic
  - 유럽 언어: gruut
- 오디오: librosa, pydub
- 유틸: tqdm, loguru, tensorboard

## 4. 모델 크기

- 언어별 모델 약 200MB
- 멀티스피커 확장·커스텀 모델은 300~500MB 수준

## 5. 언어 지원

- 영어(US/UK/Indian/Australian)
- 중국어(중영 혼합)
- 한국어
- 일본어
- 스페인어
- 프랑스어
- 일부 Fork: Malay(MS)