"""
Continuous duplex /call WebSocket.

Client:  AudioWorklet → 16 kHz mono PCM16 frames → ws.send(bytes)
Server:  Silero VAD chunks the stream into utterances. On end-of-speech:
         STT → LLM stream → per-sentence TTS → ws.send(WAV bytes).
         Status events (JSON text frames) keep the client in the loop.

Wire format
  Client → server:
    bytes  : raw PCM16 little-endian, 16 kHz, mono. Any size; we re-chunk to 512 samples.
    text   : JSON control:  {"type":"hello", system, model, engine, voice, language?}
                            {"type":"interrupt"}            (user wants to barge in)
                            {"type":"bye"}                  (end call)

  Server → client:
    text JSON :
      {"type":"ready"}
      {"type":"vad","speaking":bool}
      {"type":"partial_transcript","text":"..."}     (sent once per turn after STT)
      {"type":"assistant_token","text":"..."}        (LLM streaming)
      {"type":"assistant_done","text":"...","stats":{...}}
      {"type":"error","msg":"..."}
    bytes :
      WAV/PCM bytes — assistant audio, one blob per sentence.
      Header (first 8 bytes): b"WAV\\0" + uint32 little-endian payload length.
      Payload that follows is a complete WAV file.
"""
from __future__ import annotations

import asyncio
import io
import json
import re
import struct
import time
import traceback
import wave
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

# We import the rest lazily to keep startup cheap.

SAMPLE_RATE = 16000          # everything is 16 kHz mono int16
VAD_FRAME = 512              # Silero requires 512 samples at 16 kHz (~32 ms)
SPEECH_START_FRAMES = 3      # ~96 ms of speech to trigger turn start
SILENCE_END_FRAMES = 22      # ~700 ms of silence to end a turn
MIN_SPEECH_FRAMES = 8        # ignore <250 ms blips
MAX_TURN_FRAMES = 30 * 32    # ~30 seconds hard cap

WAV_PREFIX = b"WAV\0"

SENTENCE_END = re.compile(r"([.!?]+|[।])(?=\s|$)")


def _split_sentences(buf: str) -> tuple[list[str], str]:
    out = []
    last = 0
    for m in SENTENCE_END.finditer(buf):
        end = m.end()
        seg = buf[last:end].strip()
        if seg:
            out.append(seg)
        last = end
    return out, buf[last:]


def _frame_audio(pcm_bytes: bytes) -> bytes:
    """Return 16 kHz mono PCM16 as a WAV blob (in-memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class CallSession:
    def __init__(self, server_module):
        # server_module gives us the synth() / _llm() singletons
        self.s = server_module
        self.system_prompt = "You are Aria, a friendly voice assistant. Reply in 1-2 short sentences."
        self.model = "qwen-1.5b"
        self.engine = "piper"
        self.voice = "amy (US, female)"
        self.language: str | None = None
        self.history: list[dict] = []
        self.speaking = False                # *user* speaking
        self.assistant_speaking = False      # we are sending TTS audio
        self.cancel_event = asyncio.Event()  # set when user barges in

    def configure(self, msg: dict):
        if v := msg.get("system"): self.system_prompt = v
        if v := msg.get("model"): self.model = v
        if v := msg.get("engine"): self.engine = v
        if v := msg.get("voice"): self.voice = v
        self.language = msg.get("language") or None


def _wav_to_pcm16(wav_bytes: bytes, target_sr: int = SAMPLE_RATE) -> bytes:
    """Decode a WAV (any sr) → 16 kHz mono PCM16 bytes for VAD prebuffer / replay."""
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        # cheap linear resample
        n = int(len(data) * target_sr / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    pcm = np.clip(np.asarray(data, dtype=np.float32) * 32767, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def register_call_endpoint(app, server_module):
    """Mount the /call WS endpoint on the FastAPI app."""
    from silero_vad import load_silero_vad

    vad_model = load_silero_vad(onnx=True)

    @app.websocket("/call")
    async def call(ws: WebSocket):
        try:
            await ws.accept()
        except Exception as e:
            traceback.print_exc()
            print(f"WS accept failed: {e}", flush=True)
            return
        sess = CallSession(server_module)
        ws_closed = False
        await ws.send_text(json.dumps({"type": "ready"}))

        # Audio buffers
        leftover = b""                       # bytes that don't fill a 512-sample frame
        speech_frames: list[bytes] = []      # accumulated speech for current turn
        silence_run = 0
        speech_run = 0
        in_turn = False
        turn_lock = asyncio.Lock()           # serialize STT/LLM/TTS turns

        # Reset VAD state at start of session
        vad_model.reset_states()

        async def send_wav(wav_bytes: bytes):
            if ws_closed:
                return
            header = WAV_PREFIX + struct.pack("<I", len(wav_bytes))
            await ws.send_bytes(header + wav_bytes)

        async def run_turn(pcm: bytes):
            """STT → LLM stream → per-sentence TTS, all interruptible."""
            sess.cancel_event.clear()
            sess.assistant_speaking = True
            try:
                # ---- STT ----
                t0 = time.perf_counter()
                wav = _frame_audio(pcm)
                stt_result = await asyncio.to_thread(
                    sess.s.transcribe, wav, ".wav", sess.language
                )
                user_text = (stt_result.get("text") or "").strip()
                stt_seconds = round(time.perf_counter() - t0, 3)
                if not user_text:
                    if not ws_closed:
                        await ws.send_text(json.dumps({"type": "noop", "reason": "empty_transcript"}))
                    return
                if ws_closed:
                    return
                await ws.send_text(json.dumps({
                    "type": "partial_transcript",
                    "text": user_text,
                    "stt_seconds": stt_seconds,
                    "language": stt_result.get("language"),
                }))

                # ---- LLM streaming ----
                sess.history.append({"role": "user", "content": user_text})
                messages = [{"role": "system", "content": sess.system_prompt}, *sess.history]
                llm = sess.s._llm(sess.model)

                buffer_text = ""
                full_text = ""
                ttft = None
                tok_count = 0
                t_llm = time.perf_counter()

                # The blocking iterator wraps llama.cpp; pump it from a thread,
                # bridging into asyncio via a queue so we can react to cancel.
                queue: asyncio.Queue = asyncio.Queue()

                def producer():
                    try:
                        for chunk in llm.create_chat_completion(
                            messages=messages, temperature=0.7,
                            max_tokens=200, stream=True,
                        ):
                            delta = chunk["choices"][0].get("delta", {})
                            text = delta.get("content")
                            if text:
                                queue.put_nowait(("tok", text))
                            if sess.cancel_event.is_set():
                                queue.put_nowait(("cancel", None))
                                return
                        queue.put_nowait(("done", None))
                    except Exception as e:
                        queue.put_nowait(("err", str(e)))

                producer_task = asyncio.create_task(asyncio.to_thread(producer))

                first_audio_t = None
                while True:
                    if sess.cancel_event.is_set():
                        if not ws_closed:
                            await ws.send_text(json.dumps({"type": "interrupted"}))
                        break
                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if kind == "tok":
                        if ttft is None:
                            ttft = round(time.perf_counter() - t_llm, 3)
                        tok_count += 1
                        full_text += payload
                        buffer_text += payload
                        if ws_closed:
                            break
                        await ws.send_text(json.dumps({"type": "assistant_token", "text": payload}))
                        sentences, buffer_text = _split_sentences(buffer_text)
                        for sent in sentences:
                            if sess.cancel_event.is_set():
                                break
                            try:
                                wav_bytes, _, meta = await asyncio.to_thread(
                                    sess.s.synth, sess.engine, sess.voice, sent
                                )
                            except Exception as e:
                                await ws.send_text(json.dumps({"type": "error", "msg": f"tts: {e}"}))
                                continue
                            if first_audio_t is None:
                                first_audio_t = round(time.perf_counter() - t_llm, 3)
                            await send_wav(wav_bytes)
                    elif kind == "cancel":
                        if not ws_closed:
                            await ws.send_text(json.dumps({"type": "interrupted"}))
                        break
                    elif kind == "err":
                        if not ws_closed:
                            await ws.send_text(json.dumps({"type": "error", "msg": f"llm: {payload}"}))
                        break
                    elif kind == "done":
                        leftover_text = buffer_text.strip()
                        if leftover_text and not sess.cancel_event.is_set():
                            try:
                                wav_bytes, _, _ = await asyncio.to_thread(
                                    sess.s.synth, sess.engine, sess.voice, leftover_text
                                )
                                if first_audio_t is None:
                                    first_audio_t = round(time.perf_counter() - t_llm, 3)
                                await send_wav(wav_bytes)
                            except Exception as e:
                                await ws.send_text(json.dumps({"type": "error", "msg": f"tts: {e}"}))
                        break

                producer_task.cancel()
                if full_text:
                    sess.history.append({"role": "assistant", "content": full_text})
                total_seconds = round(time.perf_counter() - t0, 3)
                tps = round(tok_count / max(time.perf_counter() - t_llm, 1e-6), 2)
                if not ws_closed:
                    await ws.send_text(json.dumps({
                        "type": "assistant_done",
                        "text": full_text,
                        "stats": {
                            "stt_seconds": stt_seconds,
                            "ttft_seconds": ttft,
                            "first_audio_seconds": first_audio_t,
                            "tokens": tok_count,
                            "tokens_per_second": tps,
                            "total_seconds": total_seconds,
                        },
                    }))
            finally:
                sess.assistant_speaking = False

        async def receive_loop():
            nonlocal ws_closed
            nonlocal leftover, speech_frames, silence_run, speech_run, in_turn
            try:
                while True:
                    try:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            ws_closed = True
                            return
                        if "bytes" in msg and msg["bytes"] is not None:
                            leftover += msg["bytes"]
                            # process complete 512-sample (1024-byte) frames
                            while len(leftover) >= VAD_FRAME * 2:
                                frame_bytes = leftover[: VAD_FRAME * 2]
                                leftover = leftover[VAD_FRAME * 2:]
                                arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                                import torch
                                prob = float(vad_model(torch.from_numpy(arr), SAMPLE_RATE).item())
                                is_speech = prob > 0.5

                                if is_speech:
                                    speech_run += 1
                                    silence_run = 0
                                    if not in_turn and speech_run >= SPEECH_START_FRAMES:
                                        in_turn = True
                                        speech_frames = []
                                        await ws.send_text(json.dumps({"type": "vad", "speaking": True}))
                                        # barge in: cancel the assistant's current turn
                                        if sess.assistant_speaking:
                                            sess.cancel_event.set()
                                    if in_turn:
                                        speech_frames.append(frame_bytes)
                                else:
                                    if in_turn:
                                        silence_run += 1
                                        speech_frames.append(frame_bytes)
                                        if silence_run >= SILENCE_END_FRAMES:
                                            # end of turn
                                            in_turn = False
                                            speech_run = 0
                                            await ws.send_text(json.dumps({"type": "vad", "speaking": False}))
                                            if len(speech_frames) >= MIN_SPEECH_FRAMES:
                                                pcm = b"".join(speech_frames)
                                                asyncio.create_task(_safe_turn(pcm))
                                            speech_frames = []
                                            silence_run = 0
                                            vad_model.reset_states()
                                    else:
                                        speech_run = max(0, speech_run - 1)
                                # safety: cap turn length
                                if in_turn and len(speech_frames) >= MAX_TURN_FRAMES:
                                    in_turn = False
                                    pcm = b"".join(speech_frames)
                                    speech_frames = []; silence_run = 0; speech_run = 0
                                    await ws.send_text(json.dumps({"type": "vad", "speaking": False, "reason": "max_length"}))
                                    asyncio.create_task(_safe_turn(pcm))
                                    vad_model.reset_states()
                        elif "text" in msg and msg["text"] is not None:
                            try:
                                data = json.loads(msg["text"])
                            except Exception:
                                continue
                            kind = data.get("type")
                            if kind == "hello":
                                sess.configure(data)
                                await ws.send_text(json.dumps({"type": "configured", "engine": sess.engine, "voice": sess.voice, "model": sess.model}))
                            elif kind == "interrupt":
                                sess.cancel_event.set()
                            elif kind == "bye":
                                return
                    except WebSocketDisconnect:
                        ws_closed = True
                        return
                    except Exception as e:
                        traceback.print_exc()
                        try:
                            await ws.send_text(json.dumps({"type": "error", "msg": f"receive: {e}"}))
                        except Exception:
                            return
            except WebSocketDisconnect:
                return

        async def _safe_turn(pcm: bytes):
            async with turn_lock:
                try:
                    await run_turn(pcm)
                except Exception as e:
                    traceback.print_exc()
                    try:
                        await ws.send_text(json.dumps({"type": "error", "msg": f"turn: {e}"}))
                    except Exception:
                        pass

        try:
            await receive_loop()
        finally:
            ws_closed = True
            try:
                await ws.close()
            except Exception:
                pass

    return app
