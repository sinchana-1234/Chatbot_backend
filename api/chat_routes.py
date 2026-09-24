# backend/chat_routes.py
"""
    Revival Medical System Chat API Routes
    API endpoints for medical chat and query functionality with role-based access control
"""
 
import logging
import uuid
import os
import base64
import tempfile
import json
import subprocess
from typing import Dict, Any, Optional, Tuple
import asyncio  # ✅ added for optional async initialize
 
import requests
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
 
# === Original project imports (unchanged interface) ===
# Keep your original import path; the agent class name is the same.
from agents import MedicalLangChainAgent
from auth.auth import get_current_user, UserContext, get_authorized_patient_id
from response_builder import resolve_agent_output
 
logger = logging.getLogger(__name__)
 
# Create API router
router = APIRouter(prefix="/api/chat", tags=["chat"])
 
# Global variables (unchanged)
session_agents: Dict[str, MedicalLangChainAgent] = {}  # agent instances per session
 
 
# =========================
# Request/Response models
# =========================
 
class QueryRequest(BaseModel):
    query: str
    sessionId: Optional[str] = None
    patient_id: Optional[int] = None  # For medical staff to query specific patients
 
 
class QueryResponse(BaseModel):
    response: str
    sessionId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    agpChartData: Optional[Dict[str, Any]] = None
    ehba1cTirData: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    chart_data: Optional[Dict[str, Any]] = None  # present only when the answer includes a renderable chart
    suggestions: Optional[list] = None
 
 
# Voice models (robust to camelCase & snake_case)
class VoiceQueryRequest(BaseModel):
    # Accept both audioBase64 and audio_base64
    audio_base64: str = Field(..., alias="audioBase64", description="Base64 audio; may be a data URL")
    # Accept both sessionId and session_id
    session_id: Optional[str] = Field(None, alias="sessionId")
    # Accept both patientId and patient_id
    patient_id: Optional[int] = Field(None, alias="patientId")
    # Default to 'international' if omitted
    language: str = "international"  # regional | international
 
    model_config = {
        "populate_by_name": True,   # renamed from allow_population_by_field_name in Pydantic V2
        "extra": "ignore",
    }
 
    @validator("audio_base64")
    def ensure_non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("audio_base64 cannot be empty")
        return v
 
    @validator("language", pre=True, always=True)
    def normalize_language(cls, v: Optional[str]) -> str:
        v = (v or "").strip().lower()
        if v not in {"regional", "international"}:
            return "international"
        return v
 
 
class VoiceQueryResponse(BaseModel):
    response: str
    sessionId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chart_data: Optional[Dict[str, Any]] = None
    agpChartData: Optional[Dict[str, Any]] = None
    ehba1cTirData: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    transcript: Optional[str] = None
    suggestions: Optional[list] = None
 
 
# =========================
# Utilities / Helpers
# =========================
 
def _strip_data_url_prefix(b64: str) -> Tuple[str, Optional[str]]:
    """
    Accept pure base64 or data URLs like:
      data:audio/webm;codecs=opus;base64,AAAA...
    Returns: (base64_payload, mime) where mime can be None if not data URL
    """
    if b64 and b64.lower().startswith("data:"):
        # split once on comma
        header, payload = b64.split(",", 1)
        # parse MIME type e.g. data:audio/webm;codecs=opus;base64
        mime = None
        if header.startswith("data:"):
            # strip 'data:' and then take until ';' or end
            after = header[5:]
            semi = after.find(";")
            mime = after if semi == -1 else after[:semi]
        return payload, mime
    return b64, None
 
 
def _guess_suffix_from_mime(mime: Optional[str]) -> str:
    """Map common audio MIME types to file extensions."""
    if not mime:
        return ".wav"  # safest default we’ll convert to WAV anyway
    mime = mime.lower()
    if "webm" in mime:
        return ".webm"
    if "mp4" in mime or "m4a" in mime or "aac" in mime:
        return ".mp4"
    if "x-wav" in mime or "wav" in mime:
        return ".wav"
    if "x-m4a" in mime:
        return ".m4a"
    if "ogg" in mime:
        return ".ogg"
    if "mpeg" in mime or "mp3" in mime:
        return ".mp3"
    return ".wav"
 
 
def _ensure_wav(audio_path: str) -> str:
    """
    If the audio file is not WAV, transcode it to 16kHz mono WAV using ffmpeg,
    then return the path of the WAV file. Requires ffmpeg on PATH.
    """
    if audio_path.lower().endswith(".wav"):
        return audio_path
 
    wav_path = audio_path + ".wav"
    try:
        # ffmpeg -y -i input -ac 1 -ar 16000 -f wav output.wav
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return wav_path
    except Exception as e:
        logger.error(f"ffmpeg transcoding failed: {e}")
        # fallback: hand original to Whisper (may still work)
        return audio_path
 
 
def get_or_create_session_agent(session_id: Optional[str] = None, openai_api_key: Optional[str] = None) -> tuple:
    """Get existing session agent or create new one (kept compatible with new/old agent signatures)."""
    global session_agents
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in session_agents:
        try:
            # ✅ Newer agent signature prefers user_context
            try:
                session_agents[session_id] = MedicalLangChainAgent(user_context={})
            except TypeError:
                # ↩️ Fallback for older builds that expected openai_api_key
                session_agents[session_id] = MedicalLangChainAgent(openai_api_key=openai_api_key or '')
            logger.info(f"✅ Created new session agent for session: {session_id[:8]}...")
        except Exception as e:
            logger.error(f"❌ Failed to create session agent: {e}")
            return None, session_id
    return session_agents[session_id], session_id
 
 
async def _maybe_initialize(agent: Any):
    """Initialize agent if it exposes an async initialize() (keeps compatibility with older builds)."""
    try:
        init = getattr(agent, "initialize", None)
        if callable(init):
            res = init()
            if asyncio.iscoroutine(res):
                await res
    except Exception as e:
        logger.error(f"Agent initialize() failed (continuing): {e}")
 
 
def _transcribe_with_whisper(audio_bytes: bytes, mime: Optional[str]) -> str:
    """Whisper transcription, robust to webm/mp4 by transcoding to WAV first."""
    import whisper
    try:
        # 1) Save bytes with correct suffix for ffmpeg to parse
        suffix = _guess_suffix_from_mime(mime)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            f.flush()
            input_path = f.name
 
        # 2) Convert to wav (16k mono) for reliability
        wav_path = _ensure_wav(input_path)
 
        # 3) Transcribe
        model = whisper.load_model("small")  # use what you had
        result = model.transcribe(wav_path, fp16=False)  # CPU safe
        text = (result or {}).get("text", "") or ""
        return text.strip()
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise
 
 
def _transcribe_translate_with_sarvam(audio_bytes: bytes, mime: Optional[str]) -> str:
    """
    Sarvam AI STT + translate-to-English.
    - Auth: header 'api-subscription-key: <key>'
    - Endpoint: prefer /speech-to-text-translate (auto-detects input, returns English)
      Fallback to /speech-to-text if you want same-language transcription.
    """
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    sarvam_url = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text-translate").strip()
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment.")
 
    try:
        files = {
            "file": ("audio" + (_guess_suffix_from_mime(mime) or ".wav"),
                     audio_bytes,
                     (mime or "application/octet-stream"))
        }
        resp = requests.post(
            sarvam_url,
            headers={"api-subscription-key": api_key},
            files=files,
            timeout=60,
        )
        if not resp.ok:
            body = resp.text.strip()
            logger.error(f"Sarvam AI STT failed ({resp.status_code}): {body}")
            resp.raise_for_status()
 
        payload = resp.json() if resp.content else {}
        text = (payload.get("text")
                or payload.get("transcript")
                or payload.get("result")
                or "").strip()
        if not text:
            logger.error(f"Sarvam AI STT returned empty text. Raw: {payload}")
        return text
    except Exception as e:
        logger.error(f"Sarvam AI STT failed: {e}")
        raise
 
 
# =========================
# Routes
# =========================

from fastapi import Header

@router.post("/query", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    current_user: UserContext = Depends(get_current_user),
    authorization: Optional[str] = Header(default=None)
):
    """Handle medical queries with LangChain agent, session management, and role-based access control"""
    logger.info(f" Received medical query from user {current_user.user_id} (Role: {current_user.role_name}): {request.query[:100]}...")
 
    try:
        query = request.query.strip()
        session_id = request.sessionId
        requested_patient_id = request.patient_id
 
        if not query:
            logger.warning("❌ Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
 
        # Authorize patient access
        authorized_patient_id = get_authorized_patient_id(requested_patient_id, current_user)
 
        # Build context
        if current_user.role_id == 1:  # Patient
            query_with_context = f"[Patient Query - User ID: {current_user.user_id}] {query}"
        else:
            if authorized_patient_id:
                query_with_context = f"[Medical Staff Query - For Patient ID: {authorized_patient_id}] {query}"
            else:
                query_with_context = f"[Medical Staff Query - General] {query}"
 
        logger.info(
            f" Processing medical query from {current_user.role_name}: "
            f"'{query[:50]}{'...' if len(query) > 50 else ''}' | "
            f"Session: {session_id[:8] + '...' if session_id else 'NEW'}"
        )
 
        # Session agent
        session_agent, session_id = get_or_create_session_agent(session_id, os.getenv("OPENAI_API_KEY"))
 
        # ✅ initialize if needed (keeps compatibility with new agent)
        await _maybe_initialize(session_agent)
 
        try:
            if hasattr(session_agent, 'set_user_context'):
                session_agent.set_user_context({
                    'user_id': current_user.user_id,
                    'role_id': current_user.role_id,
                    'role_name': current_user.role_name,
                    'can_access_all_patients': current_user.can_access_all_patients,
                    'authorized_patient_id': authorized_patient_id,
                    'auth_token': authorization.replace("Bearer ", "").strip() if authorization else None,
                    'token': current_user.token
                })
 
            result = await session_agent.chat(query_with_context)
            logger.info(f"✅ Medical agent response generated successfully for user {current_user.user_id}")

            # One helper turns the agent turn into (text, charts). It reads
            # every deterministic-tool answer by convention and never drifts
            # between /query and /voice. See response_builder.resolve_agent_output.
            response_text, chart_data, agp_chart_data, ehba1c_tir_data, suggestions = resolve_agent_output(
                getattr(session_agent, "user_context", None), result
            )

            result_metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
            result_metadata["session_id"] = session_id
            result_metadata["conversation_length"] = len(session_agent.get_conversation_history() or [])
            result_metadata["user_role"] = current_user.role_name
            result_metadata["authorized_patient_id"] = authorized_patient_id

            return QueryResponse(
                response=response_text,
                sessionId=session_id,
                metadata=result_metadata,
                chart_data=chart_data,
                agpChartData=agp_chart_data,
                ehba1cTirData=ehba1c_tir_data,
                suggestions=suggestions,
                user_context={
                    "user_id": current_user.user_id,
                    "role_name": current_user.role_name,
                    "can_access_all_patients": current_user.can_access_all_patients,
                    "authorized_patient_id": authorized_patient_id
                }
            )
        except Exception as e:
            logger.error(f"❌ Medical session agent failed: {e}")
            raise HTTPException(status_code=500, detail=f"Medical agent error: {str(e)}")
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Medical query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
 
 
@router.post("/voice", response_model=VoiceQueryResponse)
async def handle_voice_query(
    request: VoiceQueryRequest,
    current_user: UserContext = Depends(get_current_user),
    authorization: Optional[str] = Header(default=None)
):
    """
    Voice:
      - 'regional' → Sarvam AI (configurable URL) + translate to English
      - 'international' → Whisper (transcode to WAV for reliability)
    """
    try:
        # 1) Decode base64
        raw_b64, mime = _strip_data_url_prefix(request.audio_base64)
        try:
            audio_bytes = base64.b64decode(raw_b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid audio_base64: cannot decode")
 
        if not audio_bytes or len(audio_bytes) < 2000:
            raise HTTPException(status_code=400, detail="Audio too short or empty")
 
        # 2) STT
        language = request.language  # normalized
        if language == "regional":
            transcript = _transcribe_translate_with_sarvam(audio_bytes, mime)
        else:
            transcript = _transcribe_with_whisper(audio_bytes, mime)
 
        if not transcript:
            raise HTTPException(status_code=422, detail="Failed to transcribe audio")
 
        # 3) Build query context (mirror /query)
        requested_patient_id = request.patient_id
        authorized_patient_id = get_authorized_patient_id(requested_patient_id, current_user)
 
        if current_user.role_id == 1:
            query_with_context = f"[Patient Query - User ID: {current_user.user_id}] {transcript}"
        else:
            if authorized_patient_id:
                query_with_context = f"[Medical Staff Query - For Patient ID: {authorized_patient_id}] {transcript}"
            else:
                query_with_context = f"[Medical Staff Query - General] {transcript}"
 
        # 4) Session agent
        session_agent, session_id = get_or_create_session_agent(request.session_id, os.getenv("OPENAI_API_KEY"))
 
        # ✅ initialize if needed (keeps compatibility with new agent)
        await _maybe_initialize(session_agent)
 
        # 5) User context
        if hasattr(session_agent, 'set_user_context'):
            session_agent.set_user_context({
                'user_id': current_user.user_id,
                'role_id': current_user.role_id,
                'role_name': current_user.role_name,
                'can_access_all_patients': current_user.can_access_all_patients,
                'authorized_patient_id': authorized_patient_id,
                'auth_token': authorization.replace("Bearer ", "").strip() if authorization else None,
                'token': current_user.token
            })

        # 6) Ask the agent
        result = await session_agent.chat(query_with_context)

        # Same single helper as /query -- identical response semantics, no drift.
        response_text, chart_data, agp_chart_data, ehba1c_tir_data,  suggestions  = resolve_agent_output(
            getattr(session_agent, "user_context", None), result
        )

        logger.info(f"✅ Voice query processed for user {current_user.user_id}")

        # 7) Metadata
        result_metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        result_metadata.update({
            "session_id": session_id,
            "conversation_length": len(session_agent.get_conversation_history() or []),
            "user_role": current_user.role_name,
            "authorized_patient_id": authorized_patient_id,
            "language_mode": language,
            "transcript_length": len(transcript),
        })

        return VoiceQueryResponse(
            response=response_text,
            sessionId=session_id,
            metadata=result_metadata,
            chart_data=chart_data,
            agpChartData=agp_chart_data,
            ehba1cTirData=ehba1c_tir_data,
            suggestions=suggestions,
            user_context={
                "user_id": current_user.user_id,
                "role_name": current_user.role_name,
                "can_access_all_patients": current_user.can_access_all_patients,
                "authorized_patient_id": authorized_patient_id
            },
            transcript=transcript
        )
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Voice query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
 
 
@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Revival Medical System API",
        "version": "1.0.0",
        "features": [
            "Medical Data Analysis",
            "LangChain Agent",
            "Conversation Memory",
            "Patient Health Records",
            "Voice Input (Sarvam/Whisper)"
        ],
        "endpoints": ["/api/chat/query", "/api/chat/voice", "/api/chat/sessions"],
        "status": "active"
    }
 
 
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Revival Medical System API",
        "version": "1.0.0"
    }
 
 
@router.get("/sessions")
async def get_active_sessions():
    """Get information about active sessions"""
    return {
        "active_sessions": len(session_agents),
        "session_ids": [session_id[:8] + "..." for session_id in session_agents.keys()]
    }