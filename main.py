"""
main.py
FreshMart Conversational AI - FastAPI Backend
Assignment 3 - Voice Interface Extension

Endpoints:
  GET  /health                  - Health check
  POST /session/new             - Create new session
  POST /session/reset/{id}      - Reset session
  GET  /session/{id}/state      - Get session state
  DELETE /session/{id}          - Delete session
  WS   /ws/chat/{id}            - Text streaming chat (Assignment 2)
  WS   /ws/voice/{id}           - Voice pipeline (Assignment 3)
"""

import uuid
import json
import base64
import logging
import asyncio
import threading
import queue as queue_module
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from conversation_manager import ConversationManager
from voice_manager import VoiceManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Session Stores ────────────────────────────────────────────────────────────
text_sessions:  dict[str, ConversationManager] = {}
voice_sessions: dict[str, VoiceManager]        = {}
PREWARM_SESSION_ID = "__prewarm__"


def get_or_create_text_session(session_id: str) -> ConversationManager:
    if session_id not in text_sessions:
        text_sessions[session_id] = ConversationManager(session_id=session_id)
        logger.info(f"New text session: {session_id}")
    return text_sessions[session_id]


def get_or_create_voice_session(session_id: str) -> VoiceManager:
    if session_id not in voice_sessions:
        vm = VoiceManager(session_id=session_id)
        # Reuse pre-warmed ASR/TTS models to avoid reloading
        prewarm = voice_sessions.get(PREWARM_SESSION_ID)
        if prewarm is not None and session_id != PREWARM_SESSION_ID:
            vm.asr = prewarm.asr
            vm.tts = prewarm.tts
        voice_sessions[session_id] = vm
        logger.info(f"New voice session: {session_id}")
    return voice_sessions[session_id]


# ── App Setup ─────────────────────────────────────────────────────────────────
def _prewarm_voice_models():
    """Load ASR + TTS models at startup so first user request is fast."""
    try:
        logger.info("Pre-warming voice models (ASR + TTS)...")
        get_or_create_voice_session(PREWARM_SESSION_ID)
        logger.info("Voice models pre-warmed and ready.")
    except Exception as e:
        logger.error(f"Voice model pre-warm failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FreshMart API starting up...")
    # Load voice models in background thread so startup doesn't block
    prewarm_thread = threading.Thread(target=_prewarm_voice_models, daemon=True)
    prewarm_thread.start()
    yield
    logger.info("FreshMart API shutting down...")
    text_sessions.clear()
    voice_sessions.clear()

app = FastAPI(
    title="FreshMart Conversational AI API",
    description="Assignment 3 - Voice + Text conversational AI",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "FreshMart Conversational AI",
        "text_sessions":  len(text_sessions),
        "voice_sessions": len(voice_sessions)
    }


@app.post("/session/new")
async def create_session():
    session_id = str(uuid.uuid4())
    get_or_create_text_session(session_id)
    return {"session_id": session_id}


@app.post("/session/reset/{session_id}")
async def reset_session(session_id: str):
    if session_id not in text_sessions and session_id not in voice_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_id in text_sessions:
        text_sessions[session_id].reset_session()
    if session_id in voice_sessions:
        voice_sessions[session_id].reset()
    return {"session_id": session_id, "status": "reset"}


@app.get("/session/{session_id}/state")
async def get_session_state(session_id: str):
    if session_id in text_sessions:
        return text_sessions[session_id].get_session_state()
    if session_id in voice_sessions:
        return voice_sessions[session_id].get_state()
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    found = False
    if session_id in text_sessions:
        del text_sessions[session_id]
        found = True
    if session_id in voice_sessions:
        del voice_sessions[session_id]
        found = True
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "status": "deleted"}


# ── Text WebSocket (Assignment 2) ─────────────────────────────────────────────

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"Text WebSocket connected: {session_id}")
    cm = get_or_create_text_session(session_id)

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()
            user_id = data.get("user_id", session_id)  # persistent user ID from localStorage

            if not user_message:
                await websocket.send_json({"type": "error", "data": "Empty message"})
                continue

            # Set the persistent user_id on the conversation manager
            cm.set_user_id(user_id)

            logger.info(f"[{session_id}] User: {user_message[:60]}")
            full_response = ""

            try:
                token_queue = queue_module.Queue()

                def stream_to_queue():
                    try:
                        for token in cm.stream_chat(user_message):
                            token_queue.put(token)
                    finally:
                        token_queue.put(None)

                thread = threading.Thread(target=stream_to_queue, daemon=True)
                thread.start()

                while True:
                    try:
                        token = token_queue.get(timeout=0.05)
                        if token is None:
                            break
                        full_response += token
                        await websocket.send_json({"type": "token", "data": token})
                    except queue_module.Empty:
                        await asyncio.sleep(0.01)

                thread.join()

            except Exception as e:
                logger.error(f"[{session_id}] Stream error: {e}")
                await websocket.send_json({"type": "error", "data": str(e)})
                continue

            await websocket.send_json({
                "type": "done",
                "data": "",
                "cart": cm.cart.get_summary(),
                "turn": cm.turn_count,
                "session_active": cm.is_active
            })

            if not cm.is_active:
                await websocket.send_json({
                    "type": "session_ended",
                    "data": "Session ended. Start a new session to continue."
                })

    except WebSocketDisconnect:
        logger.info(f"Text WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Text WebSocket error [{session_id}]: {e}")


# ── Voice WebSocket (Assignment 3) ────────────────────────────────────────────

@app.websocket("/ws/voice/{session_id}")
async def websocket_voice(websocket: WebSocket, session_id: str):
    """
    Voice WebSocket endpoint.

    Client -> Server:
        {"type": "audio", "data": "<base64 encoded WAV bytes>"}

    Server -> Client:
        {"type": "transcript",  "data": "what user said"}
        {"type": "token",       "data": "response word"}
        {"type": "audio",       "data": "<base64 encoded WAV bytes>"}
        {"type": "done",        "cart": {...}, "session_active": true}
        {"type": "error",       "data": "message"}
    """
    await websocket.accept()
    logger.info(f"Voice WebSocket connected: {session_id}")
    vm = get_or_create_voice_session(session_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") != "audio":
                await websocket.send_json({"type": "error", "data": "Expected type: audio"})
                continue

            # Apply persistent user_id to CRM (same as text WS)
            user_id = data.get("user_id", session_id)
            vm.conv.set_user_id(user_id)

            # Decode base64 audio
            try:
                audio_bytes = base64.b64decode(data["data"])
            except Exception:
                await websocket.send_json({"type": "error", "data": "Invalid base64 audio data"})
                continue

            logger.info(f"[{session_id}] Received audio: {len(audio_bytes)} bytes")

            # Run voice pipeline in thread to avoid blocking event loop
            event_queue = queue_module.Queue()

            def run_pipeline():
                try:
                    for event in vm.process_audio_streaming(audio_bytes):
                        event_queue.put(event)
                finally:
                    event_queue.put(None)

            thread = threading.Thread(target=run_pipeline, daemon=True)
            thread.start()

            while True:
                try:
                    event = event_queue.get(timeout=0.05)
                    if event is None:
                        break

                    if event["type"] == "audio":
                        # Base64 encode audio bytes for JSON transport
                        await websocket.send_json({
                            "type": "audio",
                            "data": base64.b64encode(event["data"]).decode("utf-8")
                        })
                    elif event["type"] == "done":
                        await websocket.send_json({
                            "type": "done",
                            "cart": event["cart"],
                            "session_active": event["session_active"]
                        })
                    else:
                        await websocket.send_json(event)

                except queue_module.Empty:
                    await asyncio.sleep(0.01)

            thread.join()

    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error [{session_id}]: {e}")
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass