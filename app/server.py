"""
Local voice-agent playground — TTS + LLM in one FastAPI service.

Endpoints
  GET  /                   -> static UI
  GET  /voices             -> {piper:[...], kokoro:[...]}
  GET  /models             -> [{id, label, tok_per_s_estimate}]
  POST /tts                -> WAV (one-shot synth, both engines)
  POST /chat               -> SSE stream of LLM tokens (no audio)
  POST /converse           -> multipart-like stream:
                              JSON status events + WAV audio per sentence (NDJSON over chunked HTTP)
"""
from __future__ import annotations

import asyncio
import io
import json
import re
import time
import wave
from pathlib import Path
from typing import AsyncIterator, Iterator, Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "voices"
MODELS_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

# ============================================================================
# TTS — Piper
# ============================================================================
from piper import PiperVoice

PIPER_VOICE_FILES: dict[str, Path] = {
    "amy (US, female)": VOICES_DIR / "en_US-amy-medium.onnx",
    "ryan (US, male)": VOICES_DIR / "en_US-ryan-medium.onnx",
    "pratham (Hindi, male)": VOICES_DIR / "hi_IN-pratham-medium.onnx",
    "priyamvada (Hindi, female)": VOICES_DIR / "hi_IN-priyamvada-medium.onnx",
}
_piper_cache: dict[str, PiperVoice] = {}

def _piper(name: str) -> PiperVoice:
    if name not in _piper_cache:
        _piper_cache[name] = PiperVoice.load(str(PIPER_VOICE_FILES[name]))
    return _piper_cache[name]

def synth_piper(text: str, voice_name: str) -> tuple[bytes, int, dict]:
    voice = _piper(voice_name)
    sr = voice.config.sample_rate
    t0 = time.perf_counter()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        voice.synthesize_wav(text, wf)
    elapsed = time.perf_counter() - t0
    wav = buf.getvalue()
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        audio_s = wf.getnframes() / wf.getframerate()
    return wav, sr, _meta("piper", voice_name, elapsed, audio_s, sr)

# ============================================================================
# TTS — Kokoro (lazy)
# ============================================================================
_kokoro_pipeline = None
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]

def _kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline

def synth_kokoro(text: str, voice_name: str) -> tuple[bytes, int, dict]:
    pipe = _kokoro()
    t0 = time.perf_counter()
    chunks = []
    for _, _, audio in pipe(text, voice=voice_name):
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        raise HTTPException(500, "Kokoro produced no audio")
    full = np.concatenate(chunks)
    elapsed = time.perf_counter() - t0
    sr = 24000
    audio_s = len(full) / sr
    buf = io.BytesIO()
    sf.write(buf, full, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), sr, _meta("kokoro", voice_name, elapsed, audio_s, sr)

def _meta(engine, voice, synth_s, audio_s, sr):
    rtf = synth_s / audio_s if audio_s > 0 else float("inf")
    return {
        "engine": engine, "voice": voice,
        "synth_seconds": round(synth_s, 3),
        "audio_seconds": round(audio_s, 3),
        "rtf": round(rtf, 3),
        "sample_rate": sr,
    }

def synth(engine: str, voice: str, text: str) -> tuple[bytes, int, dict]:
    if engine == "piper":
        if voice not in PIPER_VOICE_FILES:
            raise HTTPException(400, f"unknown piper voice: {voice}")
        return synth_piper(text, voice)
    if engine == "kokoro":
        if voice not in KOKORO_VOICES:
            raise HTTPException(400, f"unknown kokoro voice: {voice}")
        return synth_kokoro(text, voice)
    raise HTTPException(400, f"unknown engine: {engine}")

# ============================================================================
# LLM — llama.cpp (Qwen2.5)
# ============================================================================
from llama_cpp import Llama

LLM_REGISTRY = {
    "qwen-1.5b": {
        "label": "Qwen2.5-1.5B Q4 (fast, ~12 tok/s)",
        "path": str(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        "n_ctx": 4096,
    },
    "qwen-3b": {
        "label": "Qwen2.5-3B Q4 (smarter, ~6 tok/s)",
        "path": str(MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"),
        "n_ctx": 4096,
    },
}
_llm_cache: dict[str, Llama] = {}

def _llm(model_id: str) -> Llama:
    if model_id not in LLM_REGISTRY:
        raise HTTPException(400, f"unknown model: {model_id}")
    if model_id not in _llm_cache:
        cfg = LLM_REGISTRY[model_id]
        _llm_cache[model_id] = Llama(
            model_path=cfg["path"],
            n_ctx=cfg["n_ctx"],
            n_threads=4,
            n_batch=512,
            verbose=False,
        )
    return _llm_cache[model_id]

# ============================================================================
# STT — faster-whisper (lazy)
# ============================================================================
_whisper_model = None
_warmup_task = None

def _whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            "base", device="cpu", compute_type="int8",
            download_root=str(MODELS_DIR / "whisper"),
        )
    return _whisper_model

def transcribe(audio_bytes: bytes, suffix: str = ".webm", language: str | None = None) -> dict:
    import tempfile, subprocess
    # Whisper wants a decodable file; ffmpeg under the hood handles webm/wav/mp3.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes); path = f.name
    t0 = time.perf_counter()
    model = _whisper()
    segments, info = model.transcribe(
        path,
        language=language,           # None => auto-detect
        beam_size=1,                 # fastest
        vad_filter=True,             # drop silence
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text_parts = [s.text for s in segments]
    elapsed = time.perf_counter() - t0
    full = "".join(text_parts).strip()
    try: Path(path).unlink()
    except OSError: pass
    return {
        "text": full,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_seconds": round(info.duration, 3),
        "stt_seconds": round(elapsed, 3),
        "rtf": round(elapsed / info.duration, 3) if info.duration > 0 else None,
    }

# ============================================================================
# API models
# ============================================================================
class TTSRequest(BaseModel):
    engine: Literal["piper", "kokoro"]
    voice: str
    text: str

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen-1.5b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256

class ConverseRequest(ChatRequest):
    engine: Literal["piper", "kokoro"] = "kokoro"
    voice: str = "af_heart"


def _warm_default_assets():
    t0 = time.perf_counter()
    try:
        _whisper()
        _llm("qwen-1.5b")
        _piper("amy (US, female)")
        print(f"Warmup complete in {time.perf_counter() - t0:.2f}s", flush=True)
    except Exception as e:
        print(f"Warmup failed: {e}", flush=True)

# ============================================================================
# FastAPI
# ============================================================================
app = FastAPI(title="Local Voice Agent Playground")


@app.on_event("startup")
async def startup_warmup():
    global _warmup_task
    if _warmup_task is None:
        _warmup_task = asyncio.create_task(asyncio.to_thread(_warm_default_assets))

@app.get("/voices")
def voices():
    return {"piper": list(PIPER_VOICE_FILES.keys()), "kokoro": KOKORO_VOICES}

@app.get("/models")
def models():
    return [{"id": k, "label": v["label"]} for k, v in LLM_REGISTRY.items()]

@app.post("/stt")
async def stt(
    audio: UploadFile = File(...),
    language: str | None = Form(None),  # "en", "hi", or None for auto
):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    # Pick a sensible suffix from filename so ffmpeg can decode
    name = (audio.filename or "audio.webm").lower()
    suffix = ".webm"
    for ext in (".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"):
        if name.endswith(ext):
            suffix = ext; break
    return transcribe(data, suffix=suffix, language=language)

@app.post("/tts")
def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "text empty")
    if len(req.text) > 4000:
        raise HTTPException(400, "text too long (max 4000 chars)")
    wav, _, meta = synth(req.engine, req.voice, req.text)
    return Response(
        content=wav, media_type="audio/wav",
        headers={
            "X-TTS-Engine": meta["engine"], "X-TTS-Voice": meta["voice"],
            "X-TTS-Synth-Seconds": str(meta["synth_seconds"]),
            "X-TTS-Audio-Seconds": str(meta["audio_seconds"]),
            "X-TTS-RTF": str(meta["rtf"]), "X-TTS-Sample-Rate": str(meta["sample_rate"]),
            "Access-Control-Expose-Headers":
                "X-TTS-Engine,X-TTS-Voice,X-TTS-Synth-Seconds,X-TTS-Audio-Seconds,X-TTS-RTF,X-TTS-Sample-Rate",
        },
    )

# ----------------------------------------------------------------------------
# /chat — SSE token stream (no TTS)
# ----------------------------------------------------------------------------
def _stream_llm(llm: Llama, messages: list[dict], temperature: float, max_tokens: int) -> Iterator[dict]:
    """Yields events: {type:'token', text:'...'} and {type:'done', stats:{...}}."""
    t0 = time.perf_counter()
    first_tok_t = None
    n = 0
    full = []
    for chunk in llm.create_chat_completion(
        messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
    ):
        delta = chunk["choices"][0].get("delta", {})
        text = delta.get("content")
        if text:
            if first_tok_t is None:
                first_tok_t = time.perf_counter() - t0
            n += 1
            full.append(text)
            yield {"type": "token", "text": text}
    total = time.perf_counter() - t0
    yield {
        "type": "done",
        "stats": {
            "ttft_seconds": round(first_tok_t or 0.0, 3),
            "total_seconds": round(total, 3),
            "completion_tokens": n,
            "tokens_per_second": round(n / total, 2) if total > 0 else None,
            "full_text": "".join(full),
        },
    }

@app.post("/chat")
def chat(req: ChatRequest):
    llm = _llm(req.model)
    msgs = [m.model_dump() for m in req.messages]

    def gen():
        for evt in _stream_llm(llm, msgs, req.temperature, req.max_tokens):
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# ----------------------------------------------------------------------------
# /converse — LLM streams, sentences are TTS'd as soon as ready
# ----------------------------------------------------------------------------
SENTENCE_END = re.compile(r"([.!?]+|[।])(?=\s|$)")  # English + Hindi danda

def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Returns (complete_sentences, leftover)."""
    sentences = []
    last_end = 0
    for m in SENTENCE_END.finditer(buffer):
        end = m.end()
        seg = buffer[last_end:end].strip()
        if seg:
            sentences.append(seg)
        last_end = end
    return sentences, buffer[last_end:]

def _wav_b64(wav: bytes) -> str:
    import base64
    return base64.b64encode(wav).decode()

@app.post("/converse")
def converse(req: ConverseRequest):
    """
    NDJSON stream of events:
      {"type":"token","text":"..."}                          - LLM token
      {"type":"sentence","text":"...","audio_b64":"...","meta":{...}}  - TTS'd sentence
      {"type":"done","stats":{...}}                           - end
    """
    llm = _llm(req.model)
    msgs = [m.model_dump() for m in req.messages]

    def gen():
        t0 = time.perf_counter()
        first_audio_t = None
        sentence_count = 0
        buffer = ""
        full_text_parts = []
        llm_stats = None

        for evt in _stream_llm(llm, msgs, req.temperature, req.max_tokens):
            if evt["type"] == "token":
                yield json.dumps(evt, ensure_ascii=False) + "\n"
                buffer += evt["text"]
                full_text_parts.append(evt["text"])
                sentences, buffer = _split_sentences(buffer)
                for s in sentences:
                    try:
                        wav, _, meta = synth(req.engine, req.voice, s)
                    except Exception as e:
                        yield json.dumps({"type": "error", "where": "tts", "msg": str(e)}) + "\n"
                        continue
                    if first_audio_t is None:
                        first_audio_t = time.perf_counter() - t0
                    sentence_count += 1
                    yield json.dumps({
                        "type": "sentence", "text": s,
                        "audio_b64": _wav_b64(wav), "meta": meta,
                    }, ensure_ascii=False) + "\n"
            elif evt["type"] == "done":
                llm_stats = evt["stats"]

        # flush any trailing buffer (no sentence-ender)
        leftover = buffer.strip()
        if leftover:
            try:
                wav, _, meta = synth(req.engine, req.voice, leftover)
                if first_audio_t is None:
                    first_audio_t = time.perf_counter() - t0
                sentence_count += 1
                yield json.dumps({
                    "type": "sentence", "text": leftover,
                    "audio_b64": _wav_b64(wav), "meta": meta,
                }, ensure_ascii=False) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "where": "tts", "msg": str(e)}) + "\n"

        yield json.dumps({
            "type": "done",
            "stats": {
                "llm": llm_stats,
                "first_audio_seconds": round(first_audio_t, 3) if first_audio_t else None,
                "total_seconds": round(time.perf_counter() - t0, 3),
                "sentences": sentence_count,
            },
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")

from fastapi import WebSocket as _WS
@app.websocket("/calltest")
async def _calltest(ws: _WS):
    await ws.accept()
    await ws.send_text("ok")
    await ws.close()

# --- mount the /call WebSocket endpoint BEFORE static -----------------------
import sys as _sys
from . import call as _call_module
_call_module.register_call_endpoint(app, _sys.modules[__name__])

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
