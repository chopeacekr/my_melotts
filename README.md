# MeloTTS Server

빠르고 가벼운 다국어 TTS 서버 - MeloTTS 기반 HTTP API

## 🎯 주요 특징

- **⚡ 빠른 속도**: GPU 없이도 1~2초 내 음성 합성
- **🌍 다국어 지원**: 6개 언어 (한국어, 영어, 중국어, 일본어, 스페인어, 프랑스어)
- **💻 CPU 친화적**: CPU만으로도 실시간 합성 가능
- **🎭 멀티 스피커**: 언어별 다양한 화자 지원
- **🔊 고품질 음성**: 스튜디오 품질의 자연스러운 음성
- **⚡ FastAPI 기반**: RESTful API 제공

## 📋 시스템 요구사항

- **Python**: 3.11 이상
- **Package Manager**: UV
- **하드웨어**: CPU만으로 충분 (GPU 선택)
- **메모리**: 최소 4GB RAM

## 🚀 설치 방법

### 1. 프로젝트 클론
```bash
git clone <repository-url>
cd MeloTTS
```

### 2. UV를 사용한 의존성 설치
```bash
# 가상환경 생성 및 패키지 설치
uv sync
```

### 3. 주요 의존성
```toml
[project]
name = "melotts"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # 핵심
    "torch==2.3.1",              # PyTorch
    "torchaudio>=2.3.1",         # 오디오 처리
    "librosa==0.9.1",            # 오디오 분석
    "soundfile>=0.13.1",         # 오디오 I/O
    
    # G2P (Grapheme-to-Phoneme)
    "g2p-en==2.1.0",             # 영어
    "eng-to-ipa==0.0.2",         # 영어 IPA
    "g2pkk>=0.1.2",              # 한국어
    "jamo==0.4.1",               # 한글 자모 분리
    "pypinyin==0.50.0",          # 중국어 병음
    "jieba==0.42.1",             # 중국어 토크나이저
    "cn2an==0.5.22",             # 중국어 숫자 변환
    "mecab-python3==1.0.9",      # 일본어 형태소 분석
    "python-mecab-ko>=1.3.7",    # 한국어 MeCab
    "fugashi==1.3.0",            # 일본어 MeCab 래퍼
    "pykakasi==2.2.1",           # 일본어 히라가나 변환
    "unidic==1.1.0",             # 일본어 사전
    "unidic-lite==1.0.8",        # 일본어 경량 사전
    "gruut[de,es,fr]==2.2.3",    # 유럽 언어
    
    # 유틸리티
    "fastapi>=0.122.0",          # API 서버 (추가)
    "uvicorn>=0.38.0",           # ASGI 서버 (추가)
    "transformers==4.27.4",      # BERT 모델
    "loguru==0.7.2",             # 로깅
    "tqdm>=4.67.1",              # 진행 표시
    "pydub==0.25.1",             # 오디오 조작
    "anyascii==0.3.2",           # 유니코드 정규화
    "unidecode==1.3.7",          # ASCII 변환
    "cached-path>=1.8.0",        # 모델 캐싱
]
```

## 🎮 실행 방법

### 기본 실행
```bash
cd MeloTTS
uv run uvicorn tts_server:app --host 0.0.0.0 --port 8000
```

서버가 시작되면:
```
============================================================
🚀 MeloTTS Server Starting...
ℹ️  Device: cpu
============================================================
✅ Server ready to synthesize speech!
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 로깅 레벨 조정

`tts_server.py` 파일 상단:
```python
# 🎚️ 로깅 설정 (여기만 수정하세요!)
VERBOSE = True   # False: 최소 로그만
DEBUG = True     # False: 상세 정보 숨김
```

| 설정 | 용도 | 출력 |
|------|------|------|
| `VERBOSE=True, DEBUG=True` | 개발/디버깅 | 모든 상세 정보 |
| `VERBOSE=True, DEBUG=False` | 운영 | 핵심 로그만 |
| `VERBOSE=False, DEBUG=False` | 성능 테스트 | 최소 로그 |

## 📡 API 엔드포인트

### 1. Health Check

서버 상태 및 로드된 모델 확인
```bash
GET http://localhost:8000/health
```

**응답 예시:**
```json
{
  "status": "ok",
  "device": "cpu",
  "loaded_languages": ["KR", "EN"]
}
```

### 2. TTS 합성 (Base64)

텍스트를 음성으로 변환하고 Base64로 반환
```bash
POST http://localhost:8000/synthesize_base64
Content-Type: application/json

{
  "text": "안녕하세요! 멜로 TTS 한국어 모델입니다.",
  "lang": "KR",
  "speed": 1.0,
  "speaker": null
}
```

#### 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `text` | string | ✅ | - | 합성할 텍스트 |
| `lang` | string | ❌ | `"KR"` | 언어 코드 (아래 참조) |
| `speed` | float | ❌ | `1.0` | 속도 (0.5~2.0) |
| `speaker` | string | ❌ | `null` | 화자 ID (기본값: 언어별 기본 화자) |

#### 지원 언어 코드

| 코드 | 언어 | 화자 수 |
|------|------|---------|
| `KR` | 한국어 | 1 |
| `EN` | 영어 (미국) | 1 |
| `EN-US` | 영어 (미국) | 1 |
| `EN-BR` | 영어 (영국) | 1 |
| `EN-INDIA` | 영어 (인도) | 1 |
| `EN-AU` | 영어 (호주) | 1 |
| `ZH` | 중국어 (중영 혼합) | 다수 |
| `JP` | 일본어 | 1 |
| `ES` | 스페인어 | 1 |
| `FR` | 프랑스어 | 1 |

#### 응답 예시
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "mime_type": "audio/wav"
}
```

### 3. Speaker 목록 조회

특정 언어의 사용 가능한 화자 목록
```bash
GET http://localhost:8000/speakers/{lang}
```

**예시:**
```bash
GET http://localhost:8000/speakers/KR
```

**응답:**
```json
{
  "language": "KR",
  "speakers": ["KR"],
  "speaker_ids": {"KR": 0}
}
```

## 💻 사용 예시

### Python 클라이언트

#### 기본 사용
```python
import requests
import base64

response = requests.post(
    "http://localhost:8000/synthesize_base64",
    json={
        "text": "안녕하세요! 멜로 TTS 테스트입니다.",
        "lang": "KR",
        "speed": 1.0
    },
    timeout=30
)

# 오디오 저장
audio_b64 = response.json()["audio_base64"]
audio_bytes = base64.b64decode(audio_b64)

with open("output.wav", "wb") as f:
    f.write(audio_bytes)

print("✅ 음성 파일 생성 완료: output.wav")
```

#### 다국어 예시
```python
import requests
import base64

languages = [
    {"text": "안녕하세요", "lang": "KR"},
    {"text": "Hello world", "lang": "EN"},
    {"text": "你好世界", "lang": "ZH"},
    {"text": "こんにちは", "lang": "JP"},
]

for i, config in enumerate(languages):
    response = requests.post(
        "http://localhost:8000/synthesize_base64",
        json=config,
        timeout=30
    )
    
    audio_bytes = base64.b64decode(response.json()["audio_base64"])
    
    with open(f"output_{config['lang']}.wav", "wb") as f:
        f.write(audio_bytes)
    
    print(f"✅ {config['lang']}: output_{config['lang']}.wav")
```

#### 속도 조절 예시
```python
import requests
import base64

speeds = [0.5, 1.0, 1.5, 2.0]
text = "속도 테스트입니다"

for speed in speeds:
    response = requests.post(
        "http://localhost:8000/synthesize_base64",
        json={"text": text, "lang": "KR", "speed": speed},
        timeout=30
    )
    
    audio_bytes = base64.b64decode(response.json()["audio_base64"])
    
    with open(f"speed_{speed}x.wav", "wb") as f:
        f.write(audio_bytes)
    
    print(f"✅ {speed}x 속도: speed_{speed}x.wav")
```

### cURL 예시

#### 기본 합성
```bash
curl -X POST http://localhost:8000/synthesize_base64 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is MeloTTS.",
    "lang": "EN",
    "speed": 1.0
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Speaker 목록
```bash
curl http://localhost:8000/speakers/KR
```

## 📁 프로젝트 구조
```
MeloTTS/
├── tts_server.py           # FastAPI TTS 서버
├── melo/
│   ├── api.py             # MeloTTS API
│   ├── text/              # 텍스트 처리
│   └── models/            # 모델 정의
├── pyproject.toml          # 프로젝트 의존성
├── README.md              # 이 문서
└── .venv/                 # 가상환경 (자동 생성)
```

## ⚙️ 기술 상세

### 1. 모델 아키텍처

- **기반**: VITS / Bert-VITS2
- **구조**: Non-autoregressive (비자동회귀)
- **텍스트 인코더**: BERT 계열
- **Vocoder**: HiFi-GAN 계열
- **Alignment**: Self-aligned (내부 자동 학습)
- **조건**: Language ID + Speaker ID

### 2. 데이터 특성

- 다국어·멀티스피커 음성 데이터 기반
- 스튜디오 품질 중심의 고음질 음성
- 중국어 모델은 Chinese + English code-mixing 지원

### 3. 모델 크기

| 모델 | 크기 |
|------|------|
| 단일 언어 모델 | ~200MB |
| 멀티스피커 모델 | 300~500MB |

### 4. 성능 특징

| 항목 | CPU | GPU |
|------|-----|-----|
| 첫 요청 (모델 로딩) | ~3초 | ~2초 |
| 이후 요청 (캐싱됨) | ~1초 | ~0.5초 |
| 메모리 사용량 | ~2GB | ~3GB |

## 🐛 문제 해결

### 1. MeCab 에러
```
MeCab dictionary is not found
```

**해결책**:
```bash
# Ubuntu/Debian
sudo apt-get install mecab mecab-ipadic-utf8

# macOS
brew install mecab mecab-ipadic
```

### 2. 일본어 사전 에러
```
unidic not found
```

**해결책**:
```bash
python -m unidic download
```

### 3. gruut 언어팩 에러
```
Language 'de' not found
```

**해결책**:
```bash
uv pip install gruut[de,es,fr]
```

### 4. HParams 객체 에러
```
'HParams' object has no attribute 'get'
```

**해결책**: 이미 `tts_server.py`에서 처리됨 (HParams → dict 변환)

### 5. 느린 첫 요청

**원인**: 언어별 모델이 lazy loading됨

**해결책**: 정상 동작. 두 번째 요청부터 빨라짐
- 첫 요청: ~3초 (모델 로딩 + 합성)
- 이후: ~1초 (캐싱됨)

## 🔧 고급 설정

### 1. 포트 변경
```bash
uv run uvicorn tts_server:app --host 0.0.0.0 --port 9000
```

### 2. 워커 수 증가
```bash
uv run uvicorn tts_server:app --workers 4
```

### 3. 자동 재시작 (개발용)
```bash
uv run uvicorn tts_server:app --reload
```

### 4. HTTPS 활성화
```bash
uv run uvicorn tts_server:app \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

## 📊 성능 벤치마크

테스트 환경: Intel i7-12700K, 32GB RAM (CPU 모드)

| 텍스트 길이 | 첫 요청 | 이후 요청 |
|------------|---------|-----------|
| 짧음 (10자) | 2.8초 | 0.8초 |
| 보통 (50자) | 3.2초 | 1.2초 |
| 긴 글 (200자) | 4.5초 | 2.1초 |

## 🆚 다른 TTS 비교

| 항목 | MeloTTS | XTTS v2 | Google Cloud TTS |
|------|---------|---------|------------------|
| **속도** | 🚀 매우 빠름 (1~2초) | 🐢 느림 (8~10초) | ⚡ 빠름 (<1초) |
| **화자 복제** | ❌ 불가 | ✅ 가능 | ❌ 불가 |
| **CPU 친화적** | ✅ 매우 우수 | ❌ GPU 권장 | ❌ 클라우드 |
| **품질** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **비용** | 🆓 무료 (로컬) | 🆓 무료 (로컬) | 💰 종량제 |
| **오프라인** | ✅ 가능 | ✅ 가능 | ❌ 불가 |
| **언어 수** | 6개 | 14개+ | 40개+ |
| **상업적 이용** | ✅ MIT | ✅ MPL 2.0 | ⚠️ 약관 확인 |

## 📝 라이선스

MIT License

## 🤝 기여

이슈 제보 및 풀 리퀘스트를 환영합니다!

### 기여 방법
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 참고 자료

- [MeloTTS Original GitHub](https://github.com/myshell-ai/MeloTTS)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🙋 FAQ

**Q: GPU가 없어도 사용할 수 있나요?**  
A: 네! MeloTTS는 CPU만으로도 실시간 합성이 가능하도록 최적화되었습니다.

**Q: 화자 복제가 가능한가요?**  
A: 아니요. 화자 복제가 필요하면 XTTS v2를 사용하세요.

**Q: 모든 언어를 동시에 사용할 수 있나요?**  
A: 네. 각 언어 모델은 첫 요청 시 자동으로 로드되고 캐싱됩니다.

**Q: 중국어-영어 코드 믹싱이란?**  
A: 중국어 문장에 영어 단어가 섞여도 자연스럽게 합성됩니다.
예: "今天的weather很好" → 자연스럽게 발음

**Q: 상업적으로 사용할 수 있나요?**  
A: MIT 라이선스로 상업적 사용 가능합니다.

## 📧 문의

- **이슈 제보**: [GitHub Issues](링크)
- **이메일**: chopeacekr@gmail.com
- **디스코드**: [커뮤니티 링크]

## 🎉 감사의 글

- MyShell AI (Original MeloTTS)
- VITS Contributors
- Bert-VITS2 Developers
- FastAPI Community

---

**Version**: 0.1.0  
**Last Updated**: 2024-11-26  
**Made with** ❤️ **by Peace Cho**