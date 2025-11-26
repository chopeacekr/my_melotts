import os
import base64
from tempfile import NamedTemporaryFile
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from melo.api import TTS as MeloTTS

# ================================
# 🎚️ 로깅 설정 (여기만 수정하세요!)
# ================================
VERBOSE = True  # False로 변경하면 최소 로그만 출력
DEBUG = True    # False로 변경하면 상세 정보 숨김

device = "cpu"  # MeloTTS는 주로 CPU 사용

def log(message: str, level: str = "INFO"):
    """로깅 함수"""
    if not VERBOSE:
        return
    
    emoji_map = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️ ",
        "START": "🚀",
        "REQUEST": "📨",
        "PROCESS": "🔊",
        "FILE": "📁",
        "CLEAN": "🗑️ ",
        "TIME": "⏱️ ",
    }
    
    emoji = emoji_map.get(level, "  ")
    print(f"{emoji} {message}")

def log_debug(message: str):
    """디버그 로깅 (DEBUG=True일 때만 출력)"""
    if DEBUG and VERBOSE:
        print(f"   {message}")

def log_separator(char="=", length=60):
    """구분선 출력"""
    if VERBOSE:
        print(char * length)

# 서버 시작
log_separator()
log("MeloTTS Server Starting...", "START")
log(f"Device: {device}", "INFO")
log_separator()

# MeloTTS 모델 캐시 (언어별로 lazy loading)
melo_models = {}

def get_melo_model(lang: str) -> MeloTTS:
    """언어별 MeloTTS 모델 로드 (캐싱)"""
    if lang not in melo_models:
        log(f"Loading MeloTTS model for language: {lang}", "INFO")
        start_time = time.time()
        try:
            model = MeloTTS(language=lang, device=device)
            elapsed = time.time() - start_time
            log(f"Model '{lang}' loaded successfully in {elapsed:.2f}s", "SUCCESS")
            melo_models[lang] = model
        except Exception as e:
            log(f"Failed to load model '{lang}': {e}", "ERROR")
            raise
    else:
        log_debug(f"Using cached model for '{lang}'")
    
    return melo_models[lang]

log("Server ready to synthesize speech!", "SUCCESS")
log_separator()

app = FastAPI(title="MeloTTS Server")

class TTSRequest(BaseModel):
    text: str
    lang: str = "KR"  # KR, EN, JP, FR, ES, ZH 등
    speed: float = 1.0
    speaker: str | None = None  # MeloTTS speaker ID (선택)

class TTSResponse(BaseModel):
    audio_base64: str
    mime_type: str = "audio/wav"

@app.get("/health")
def health():
    log("Health check requested", "INFO")
    loaded_langs = list(melo_models.keys())
    log_debug(f"Loaded models: {loaded_langs if loaded_langs else 'None'}")
    return {
        "status": "ok",
        "device": device,
        "loaded_languages": loaded_langs
    }

@app.post("/synthesize_base64", response_model=TTSResponse)
def synthesize_base64(req: TTSRequest):
    request_start = time.time()
    
    if VERBOSE:
        print()  # 빈 줄
        log_separator()
    
    log("New TTS Request", "REQUEST")
    log_debug(f"Language: {req.lang}")
    log_debug(f"Speed: {req.speed}")
    log_debug(f"Speaker: {req.speaker or 'default'}")
    log_debug(f"Text length: {len(req.text)} chars")
    if DEBUG:
        preview = req.text[:100] + ("..." if len(req.text) > 100 else "")
        log_debug(f"Text preview: {preview}")
    
    text = req.text.strip()
    if not text:
        log("Text is empty", "ERROR")
        raise HTTPException(status_code=400, detail="Text is empty")
    
    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        out_path = tmp_out.name
    log_debug(f"Temp output: {out_path}")
    
    model_load_time = 0
    tts_elapsed = 0
    
    try:
        # 모델 로드
        model_start = time.time()
        model = get_melo_model(req.lang)
        model_load_time = time.time() - model_start
        
        if model_load_time > 0.1:  # 새로 로드했을 때만 표시
            log_debug(f"Model load time: {model_load_time:.2f}s")
        
        # Speaker ID 결정
        speaker_ids = model.hps.data.spk2id
        
        # HParams 객체를 dict로 변환
        if hasattr(speaker_ids, '__dict__'):
            speaker_dict = vars(speaker_ids)
        elif hasattr(speaker_ids, 'items'):
            speaker_dict = dict(speaker_ids.items())
        else:
            # 딕셔너리인 경우
            speaker_dict = speaker_ids
        
        log_debug(f"Available speakers: {list(speaker_dict.keys())}")
        
        if req.speaker and req.speaker in speaker_dict:
            speaker_id = speaker_dict[req.speaker]
            log_debug(f"Using speaker: {req.speaker} (ID: {speaker_id})")
        else:
            # 기본 speaker: 언어 코드와 동일한 이름 또는 첫 번째
            if req.lang in speaker_dict:
                speaker_id = speaker_dict[req.lang]
                default_name = req.lang
            else:
                speaker_id = list(speaker_dict.values())[0]
                default_name = list(speaker_dict.keys())[0]
            log_debug(f"Using default speaker: {default_name} (ID: {speaker_id})")
        
        # TTS 생성
        tts_start = time.time()
        log("Synthesizing speech...", "PROCESS")
        
        model.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            output_path=out_path,
            speed=req.speed,
        )
        
        tts_elapsed = time.time() - tts_start
        log(f"Synthesis completed ({tts_elapsed:.2f}s)", "SUCCESS")
        
        # 파일 읽기 및 base64 인코딩
        encode_start = time.time()
        log_debug("Encoding to base64...")
        
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        
        if DEBUG:
            file_size_kb = len(audio_bytes) / 1024
            log_debug(f"Audio size: {file_size_kb:.2f} KB")
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        encode_elapsed = time.time() - encode_start
        log_debug(f"Encoded in {encode_elapsed:.2f}s")
        
        # 총 처리 시간
        total_elapsed = time.time() - request_start
        log(f"Request completed ({total_elapsed:.2f}s)", "SUCCESS")
        
        if DEBUG:
            log_debug(f"Breakdown:")
            if model_load_time > 0.1:
                log_debug(f"  Model load: {model_load_time:.2f}s")
            log_debug(f"  Synthesis: {tts_elapsed:.2f}s")
            log_debug(f"  Encoding: {encode_elapsed:.2f}s")
        
        if VERBOSE:
            log_separator()
        
        return TTSResponse(audio_base64=audio_b64, mime_type="audio/wav")
        
    except Exception as e:
        error_elapsed = time.time() - request_start
        log(f"ERROR after {error_elapsed:.2f}s: {str(e)}", "ERROR")
        
        if DEBUG:
            import traceback
            print("\n📋 Full traceback:")
            traceback.print_exc()
        
        if VERBOSE:
            log_separator()
        
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}") from e
        
    finally:
        # Cleanup
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
                log_debug(f"Cleaned: output file")
        except Exception as e:
            log_debug(f"Failed to remove output: {e}")


@app.get("/speakers/{lang}")
def get_speakers(lang: str):
    """특정 언어의 사용 가능한 speaker 목록 반환"""
    log(f"Speaker list requested for language: {lang}", "INFO")
    try:
        model = get_melo_model(lang)
        speaker_ids = model.hps.data.spk2id
        
        # HParams 객체를 dict로 변환
        if hasattr(speaker_ids, '__dict__'):
            speaker_dict = vars(speaker_ids)
        elif hasattr(speaker_ids, 'items'):
            speaker_dict = dict(speaker_ids.items())
        else:
            speaker_dict = speaker_ids
        
        return {
            "language": lang,
            "speakers": list(speaker_dict.keys()),
            "speaker_ids": speaker_dict
        }
    except Exception as e:
        log(f"Failed to get speakers for '{lang}': {e}", "ERROR")
        raise HTTPException(status_code=400, detail=f"Invalid language: {lang}")