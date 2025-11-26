import os
import uuid
import tempfile
import base64

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from melo.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", device)

app = FastAPI(title="MeloTTS TTS Server")

_models: dict[str, TTS] = {}
_speaker_ids: dict[str, object] = {}


class TTSRequest(BaseModel):
    text: str
    lang: str = "KR"
    speaker: str | None = None
    speed: float = 1.0


class TTSBase64Response(BaseModel):
    audio_base64: str
    mime_type: str = "audio/wav"


def get_model(lang: str):
    lang = lang.upper()
    if lang not in _models:
        print(f"[INFO] Loading MeloTTS model for language={lang}")
        model = TTS(language=lang, device=device)
        _models[lang] = model
        _speaker_ids[lang] = model.hps.data.spk2id
    return _models[lang], _speaker_ids[lang]


def choose_speaker_key(speakers_obj, lang: str, req_speaker: str | None) -> str:
    """
    speakers_obj: model.hps.data.spk2id (dict 비슷한 객체)
    lang: "KR", "EN" 등
    req_speaker: 사용자가 지정한 speaker 이름 (없으면 None)
    """
    # 1) dict처럼 keys()가 있다면 우선 활용
    if hasattr(speakers_obj, "keys"):
        keys = list(speakers_obj.keys())
    else:
        # keys()가 없으면 dir()에서 언더스코어 없는 속성만 필터
        keys = [k for k in dir(speakers_obj) if not k.startswith("_")]

    if not keys:
        raise RuntimeError("No speakers found in spk2id")

    # 사용자가 speaker를 명시한 경우
    if req_speaker is not None and req_speaker in keys:
        return req_speaker

    # 언어별 기본 speaker 우선 시도
    lang = lang.upper()
    if lang in keys:
        return lang

    # 영어인 경우 EN-Default 같은 이름이 있을 수 있음
    if lang == "EN":
        if "EN-Default" in keys:
            return "EN-Default"

    # 그래도 못 찾으면 첫 번째 키
    return keys[0]


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    lang = req.lang.upper()
    model, speakers = get_model(lang)

    spk_key = choose_speaker_key(speakers, lang, req.speaker)
    spk_id = speakers[spk_key]
    print(f"[TTS] lang={lang}, speaker={spk_key}, speed={req.speed}")

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    model.tts_to_file(text, spk_id, path, speed=req.speed)

    filename = f"{lang}_{uuid.uuid4().hex}.wav"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return FileResponse(path, media_type="audio/wav", filename=filename, headers=headers)


@app.post("/synthesize_base64", response_model=TTSBase64Response)
def synthesize_base64(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    lang = req.lang.upper()
    model, speakers = get_model(lang)

    spk_key = choose_speaker_key(speakers, lang, req.speaker)
    spk_id = speakers[spk_key]
    print(f"[TTS-BASE64] lang={lang}, speaker={spk_key}, speed={req.speed}")

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    model.tts_to_file(text, spk_id, path, speed=req.speed)

    with open(path, "rb") as f:
        audio_bytes = f.read()
    os.remove(path)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return TTSBase64Response(audio_base64=audio_b64, mime_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok"}
