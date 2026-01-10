#!/usr/bin/env python3
"""
LiveKit AI Voice Agent with Vision & Memory
- Receives audio from LiveKit room participants
- Transcribes with Whisper STT (local)
- Generates response with Ollama LLM
- Sends TTS response back via LiveKit
- Vision: Can describe what camera sees on request
- Memory: ChromaDB for persistent conversation memory

Usage:
  python livekit_agent.py                     # Connect to room
  python livekit_agent.py --room my-room      # Specify room name
  python livekit_agent.py --config custom.yml # Use custom config
"""
import asyncio
import argparse
import os
import logging
import base64
import uuid
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import aiohttp
import yaml

from livekit import rtc, api

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to load ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


# ============================================================
# Configuration Loader
# ============================================================

class Config:
    """Configuration manager - loads from YAML file"""
    
    def __init__(self, config_path: str = "config.yml"):
        self.config_path = Path(config_path)
        self._config = {}
        self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            self._config = {}
    
    def get(self, *keys, default=None):
        """Get nested config value: config.get('llm', 'model', default='gpt-4')"""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
    
    # LiveKit
    @property
    def livekit_url(self):
        return os.getenv("LIVEKIT_URL") or self.get("livekit", "url", default="ws://localhost:7880")
    
    @property
    def livekit_api_key(self):
        return os.getenv("LIVEKIT_API_KEY") or self.get("livekit", "api_key", default="devkey")
    
    @property
    def livekit_api_secret(self):
        return os.getenv("LIVEKIT_API_SECRET") or self.get("livekit", "api_secret", default="secret")
    
    @property
    def room_name(self):
        return self.get("livekit", "room", default="vchat-room")
    
    @property
    def agent_identity(self):
        return self.get("livekit", "agent_identity", default="ai-agent")
    
    @property
    def agent_name(self):
        return self.get("livekit", "agent_name", default="AI Voice Agent")
    
    # System
    @property
    def system_prompt(self):
        return self.get("system", "prompt", default="You are a helpful voice assistant. Keep responses short.")
    
    @property
    def greeting(self):
        return self.get("system", "greeting", default="Hello! I'm your voice assistant. How can I help?")
    
    @property
    def vision_triggers(self):
        return self.get("system", "vision_triggers", default=[
            "what do you see", "what can you see", "look at this", "describe what you see"
        ])
    
    # STT
    @property
    def whisper_model(self):
        return os.getenv("WHISPER_MODEL") or self.get("stt", "model", default="small")
    
    @property
    def whisper_language(self):
        return self.get("stt", "language", default="en")
    
    @property
    def stt_use_gpu(self):
        return self.get("stt", "use_gpu", default=True)
    
    # LLM
    @property
    def ollama_base_url(self):
        return os.getenv("OLLAMA_BASE_URL") or self.get("llm", "base_url", default="http://localhost:11434")
    
    @property
    def text_model(self):
        return os.getenv("OLLAMA_MODEL") or self.get("llm", "text_model", default="qwen2.5:0.5b")
    
    @property
    def vision_model(self):
        return os.getenv("OLLAMA_VISION_MODEL") or self.get("llm", "vision_model", default="qwen3-vl:2b-instruct-bf16")
    
    @property
    def llm_temperature(self):
        return self.get("llm", "temperature", default=0.7)
    
    @property
    def llm_max_tokens(self):
        return self.get("llm", "max_tokens", default=256)
    
    # TTS
    @property
    def tts_voice(self):
        return os.getenv("KOKORO_VOICE") or self.get("tts", "voice", default="af_heart")
    
    @property
    def tts_speed(self):
        return self.get("tts", "speed", default=1.0)
    
    @property
    def tts_sample_rate(self):
        return self.get("tts", "sample_rate", default=24000)
    
    # Audio
    @property
    def sample_rate(self):
        return self.get("audio", "sample_rate", default=48000)
    
    @property
    def channels(self):
        return self.get("audio", "channels", default=1)
    
    @property
    def chunk_size(self):
        return self.get("audio", "chunk_size", default=960)
    
    @property
    def vad_threshold(self):
        return self.get("audio", "vad", "threshold", default=0.02)
    
    @property
    def vad_barge_in_threshold(self):
        return self.get("audio", "vad", "barge_in_threshold", default=0.05)
    
    @property
    def vad_silence_duration(self):
        return self.get("audio", "vad", "silence_duration", default=1.0)
    
    @property
    def vad_min_speech_duration(self):
        return self.get("audio", "vad", "min_speech_duration", default=0.5)
    
    # Vision
    @property
    def vision_enabled(self):
        return self.get("vision", "enabled", default=True)
    
    @property
    def camera_index(self):
        return self.get("vision", "camera_index", default=0)
    
    @property
    def vision_width(self):
        return self.get("vision", "width", default=640)
    
    @property
    def vision_height(self):
        return self.get("vision", "height", default=480)
    
    @property
    def vision_prompt(self):
        return self.get("vision", "prompt", default="Describe what you see in this image concisely.")
    
    @property
    def save_captures(self):
        return self.get("vision", "save_captures", default=False)
    
    @property
    def captures_dir(self):
        return self.get("vision", "captures_dir", default="./captures")
    
    # Memory (ChromaDB)
    @property
    def memory_enabled(self):
        return self.get("memory", "enabled", default=True) and CHROMADB_AVAILABLE
    
    @property
    def memory_persist_dir(self):
        return self.get("memory", "persist_dir", default="./memory_db")
    
    @property
    def memory_collection_name(self):
        return self.get("memory", "collection_name", default="vchat_memory")
    
    @property
    def memory_top_k(self):
        return self.get("memory", "top_k", default=3)
    
    @property
    def memory_min_similarity(self):
        return self.get("memory", "min_similarity", default=0.5)
    
    @property
    def memory_store_user(self):
        return self.get("memory", "store_user_messages", default=True)
    
    @property
    def memory_store_assistant(self):
        return self.get("memory", "store_assistant_messages", default=False)
    
    # Logging
    @property
    def log_level(self):
        return self.get("logging", "level", default="INFO")
    
    @property
    def log_format(self):
        return self.get("logging", "format", default="%(asctime)s - %(levelname)s - %(message)s")


# Global config instance
config = Config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format=config.log_format
)
logger = logging.getLogger("vchat-agent")


# ============================================================
# ChromaDB Memory
# ============================================================

class Memory:
    """Persistent memory using ChromaDB for context retrieval"""
    
    def __init__(self):
        self.enabled = config.memory_enabled
        self.client = None
        self.collection = None
        
        if self.enabled:
            self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            persist_dir = Path(config.memory_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name=config.memory_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            count = self.collection.count()
            logger.info(f"✓ ChromaDB memory initialized ({count} memories)")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.enabled = False
    
    def store(self, text: str, role: str = "user", metadata: dict = None):
        """Store a message in memory"""
        if not self.enabled or not self.collection:
            return
        
        # Skip short messages
        if len(text.strip()) < 10:
            return
        
        try:
            doc_id = str(uuid.uuid4())
            doc_metadata = {
                "role": role,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            self.collection.add(
                documents=[text],
                ids=[doc_id],
                metadatas=[doc_metadata]
            )
            logger.debug(f"💾 Stored memory: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
    
    def retrieve(self, query: str, n_results: int = None) -> list[dict]:
        """Retrieve relevant memories for a query"""
        if not self.enabled or not self.collection:
            return []
        
        try:
            n_results = n_results or config.memory_top_k
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            memories = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= config.memory_min_similarity:
                        memories.append({
                            "text": doc,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity
                        })
            
            if memories:
                logger.info(f"🧠 Retrieved {len(memories)} relevant memories")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def get_context(self, query: str) -> str:
        """Get formatted context from relevant memories"""
        memories = self.retrieve(query)
        
        if not memories:
            return ""
        
        # Sort by similarity (highest first)
        memories.sort(key=lambda x: x['similarity'], reverse=True)
        
        context_parts = ["[Relevant memories from past conversations - USE this information:]"]
        for mem in memories:
            text = mem['text']
            # Highlight important info
            context_parts.append(f"- {text}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear all memories"""
        if self.enabled and self.collection:
            try:
                # Delete and recreate collection
                self.client.delete_collection(config.memory_collection_name)
                self.collection = self.client.create_collection(
                    name=config.memory_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("🧹 Memory cleared")
            except Exception as e:
                logger.error(f"Failed to clear memory: {e}")


# ============================================================
# Whisper STT
# ============================================================

class WhisperSTT:
    """Local Whisper-based Speech-to-Text"""
    
    def __init__(self):
        self.model_name = config.whisper_model
        self.model = None
        self.device = "cuda" if (config.stt_use_gpu and torch.cuda.is_available()) else "cpu"
    
    def load(self):
        if self.model is None:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("✓ Whisper model loaded")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text"""
        self.load()
        
        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples).astype(np.float32)
        
        # Normalize
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Transcribe
        result = self.model.transcribe(
            audio_data,
            language=config.whisper_language,
            fp16=self.device == "cuda",
            temperature=config.get("stt", "temperature", default=0.0),
            no_speech_threshold=config.get("stt", "no_speech_threshold", default=0.5),
        )
        
        return result.get("text", "").strip()


# ============================================================
# Ollama LLM (with Vision & Memory support)
# ============================================================

class OllamaLLM:
    """Ollama-based Language Model with Vision and Memory support"""
    
    def __init__(self, memory: Memory = None):
        self.text_model = config.text_model
        self.vision_model = config.vision_model
        self.base_url = config.ollama_base_url
        self.conversation_history = []
        self.memory = memory
    
    def _build_prompt(self, user_message: str, memory_context: str = "") -> str:
        """Build prompt with system context, memory, and history"""
        prompt = f"System: {config.system_prompt}\n\n"
        
        # Add memory context if available
        if memory_context:
            prompt += f"{memory_context}\n\n"
        
        # Add recent history (last 4 exchanges)
        for msg in self.conversation_history[-8:]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        prompt += f"User: {user_message}\nAssistant:"
        return prompt
    
    async def generate(self, text: str) -> str:
        """Generate text response with memory retrieval"""
        # Retrieve relevant memories
        memory_context = ""
        if self.memory:
            memory_context = self.memory.get_context(text)
            if memory_context:
                logger.info(f"🧠 Memory context found:\n{memory_context}")
            else:
                logger.debug("No relevant memories found")
        
        self.conversation_history.append({"role": "user", "content": text})
        
        # Store user message in memory
        if self.memory and config.memory_store_user:
            self.memory.store(text, role="user")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.text_model,
                        "prompt": self._build_prompt(text, memory_context),
                        "stream": False,
                        "options": {
                            "temperature": config.llm_temperature,
                            "num_predict": config.llm_max_tokens,
                        }
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response = data.get("response", "").strip()
                        self.conversation_history.append({"role": "assistant", "content": response})
                        
                        # Store assistant response in memory
                        if self.memory and config.memory_store_assistant:
                            self.memory.store(response, role="assistant")
                        
                        return response
                    else:
                        logger.error(f"Ollama error: {resp.status}")
                        return "Sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "Sorry, I couldn't connect to the language model."
    
    async def generate_with_vision(self, text: str, image_base64: str) -> str:
        """Generate response with image analysis"""
        # 30 second timeout for vision (it can be slow)
        timeout = aiohttp.ClientTimeout(total=30)
        
        logger.info(f"🔍 Sending image to {self.vision_model}...")
        logger.info(f"   Image size: {len(image_base64)} bytes")
        logger.info(f"   Prompt: {config.vision_prompt[:50]}...")
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                request_data = {
                    "model": self.vision_model,
                    "prompt": f"{config.vision_prompt}\n\nUser asked: {text}",
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": config.llm_temperature,
                        "num_predict": config.llm_max_tokens,
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data
                ) as resp:
                    logger.info(f"📡 Vision API response: {resp.status}")
                    
                    if resp.status == 200:
                        data = await resp.json()
                        response = data.get("response", "").strip()
                        
                        if not response:
                            logger.warning("⚠️ Vision model returned empty response")
                            return "I can see the image but couldn't describe it."
                        
                        logger.info(f"✓ Vision response: {response[:100]}...")
                        
                        self.conversation_history.append({"role": "user", "content": f"[Showed image] {text}"})
                        self.conversation_history.append({"role": "assistant", "content": response})
                        return response
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ Vision model error: {resp.status}")
                        logger.error(f"   Response: {error_text[:200]}")
                        
                        if "not found" in error_text.lower():
                            return f"Vision model '{self.vision_model}' not found. Run: ollama pull {self.vision_model}"
                        
                        return "Sorry, I couldn't analyze the image."
                        
        except asyncio.TimeoutError:
            logger.error("⏱️ Vision request timed out (30s)")
            return "Sorry, image analysis took too long. Try again."
        except aiohttp.ClientError as e:
            logger.error(f"❌ Connection error: {e}")
            return "Sorry, couldn't connect to the vision model. Is Ollama running?"
        except Exception as e:
            logger.error(f"❌ Vision Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, I couldn't process the image."
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# ============================================================
# Kokoro TTS
# ============================================================

class KokoroTTS:
    """Kokoro Text-to-Speech"""
    
    def __init__(self):
        self.voice = config.tts_voice
        self.speed = config.tts_speed
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self):
        if self.pipeline is None:
            try:
                from kokoro import KPipeline
                self.pipeline = KPipeline(
                    lang_code='en-us',
                    repo_id='hexgrad/Kokoro-82M',
                    device=self.device
                )
                logger.info(f"✓ Kokoro TTS loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load Kokoro: {e}")
                raise
    
    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio numpy array (24kHz, int16)"""
        self.load()
        
        results = self.pipeline(text, voice=self.voice, speed=self.speed)
        
        audio_chunks = []
        for result in results:
            if result.audio is not None:
                audio_chunks.append(result.audio)
        
        if not audio_chunks:
            return None
        
        audio = torch.cat(audio_chunks, dim=-1)
        audio_np = audio.cpu().numpy()
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        
        return audio_int16


# ============================================================
# Vision (Camera Capture)
# ============================================================

class VisionCapture:
    """Camera capture for vision capabilities"""
    
    def __init__(self, auto_init: bool = False):
        self.enabled = config.vision_enabled
        self.camera = None
        self.camera_index = config.camera_index
        self.cv2 = None
        
        if auto_init and self.enabled:
            self.init_camera()
    
    def init_camera(self):
        """Initialize camera - call this early to have camera ready"""
        if self.camera is not None:
            return True
        
        if not self.enabled:
            logger.warning("Vision is disabled in config")
            return False
        
        try:
            import cv2
            self.cv2 = cv2
            
            logger.info(f"📷 Opening camera {self.camera_index}...")
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"❌ Could not open camera {self.camera_index}")
                self.enabled = False
                return False
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.vision_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.vision_height)
            
            # Warm up camera (first few frames are often dark)
            for _ in range(5):
                self.camera.read()
            
            # Get actual resolution
            actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"✓ Camera ready: {actual_w}x{actual_h}")
            return True
            
        except ImportError:
            logger.error("❌ OpenCV (cv2) not installed. Run: pip install opencv-python")
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"❌ Camera initialization failed: {e}")
            self.enabled = False
            return False
    
    def capture(self) -> str | None:
        """Capture image and return as base64"""
        if not self.enabled:
            logger.warning("Vision capture called but vision is disabled")
            return None
        
        # Initialize if not already done
        if self.camera is None:
            if not self.init_camera():
                return None
        
        if not self.camera.isOpened():
            logger.error("Camera is not opened")
            return None
        
        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret or frame is None:
                logger.error("Failed to capture frame from camera")
                return None
            
            logger.info(f"📸 Captured frame: {frame.shape[1]}x{frame.shape[0]}")
            
            # Save if configured
            if config.save_captures:
                captures_dir = Path(config.captures_dir)
                captures_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = captures_dir / f"capture_{timestamp}.jpg"
                self.cv2.imwrite(str(save_path), frame)
                logger.info(f"💾 Saved capture to {save_path}")
            
            # Encode to base64
            encode_params = [self.cv2.IMWRITE_JPEG_QUALITY, 85]
            success, buffer = self.cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                logger.error("Failed to encode image to JPEG")
                return None
            
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info(f"✓ Image encoded: {len(image_base64)} bytes")
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def release(self):
        """Release camera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            logger.info("📷 Camera released")


# ============================================================
# Voice Activity Detection
# ============================================================

class SimpleVAD:
    """Simple Voice Activity Detection using RMS"""
    
    def __init__(self):
        self.threshold = config.vad_threshold
        self.barge_in_threshold = config.vad_barge_in_threshold
        self.silence_duration = config.vad_silence_duration
        self.min_speech_duration = config.vad_min_speech_duration
        
        self.silence_samples = 0
        self.speech_samples = 0
        self.is_speaking = False
    
    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[bool, bool]:
        """
        Process audio chunk and return (is_speech, end_of_speech)
        """
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > self.threshold:
            self.is_speaking = True
            self.silence_samples = 0
            self.speech_samples += len(audio)
            return True, False
        else:
            if self.is_speaking:
                self.silence_samples += len(audio)
                silence_time = self.silence_samples / sample_rate
                speech_time = self.speech_samples / sample_rate
                
                if silence_time >= self.silence_duration:
                    self.is_speaking = False
                    self.silence_samples = 0
                    
                    # Only trigger if enough speech was captured
                    if speech_time >= self.min_speech_duration:
                        self.speech_samples = 0
                        return False, True  # End of speech
                    
                    self.speech_samples = 0
                    return False, False
                
                return True, False  # Still in speech (brief pause)
            
            return False, False
    
    def is_barge_in(self, audio: np.ndarray) -> bool:
        """Check if audio is strong enough for barge-in"""
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > self.barge_in_threshold
    
    def reset(self):
        self.silence_samples = 0
        self.speech_samples = 0
        self.is_speaking = False


# ============================================================
# Audio Processor
# ============================================================

class AudioProcessor:
    """Processes audio: STT → LLM → TTS (with Vision & Memory support)"""
    
    def __init__(self):
        logger.info("Initializing AudioProcessor...")
        
        self.memory = Memory()
        self.stt = WhisperSTT()
        self.llm = OllamaLLM(memory=self.memory)
        self.tts = KokoroTTS()
        self.vad = SimpleVAD()
        
        # Initialize camera early so it's ready for vision requests
        self.vision = VisionCapture(auto_init=True)
        
        self.audio_buffer = []
        self.is_processing = False
        self.tts_playing = False
        
        logger.info("✓ AudioProcessor ready")
    
    def _check_vision_trigger(self, text: str) -> bool:
        """Check if text contains vision trigger phrase"""
        text_lower = text.lower()
        for trigger in config.vision_triggers:
            if trigger in text_lower:
                logger.info(f"👁️ Vision trigger matched: '{trigger}' in '{text}'")
                return True
        logger.debug(f"No vision trigger in: '{text}'")
        return False
    
    def add_audio(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Add audio chunk, return True if ready to process"""
        # Check for barge-in
        if self.tts_playing:
            if self.vad.is_barge_in(audio):
                logger.info("🛑 Barge-in detected!")
                self.tts_playing = False
                self.vad.reset()
                self.audio_buffer = [audio]
                return False
            return False
        
        if self.is_processing:
            return False
        
        is_speech, end_of_speech = self.vad.process(audio, sample_rate)
        
        if is_speech or self.vad.is_speaking:
            self.audio_buffer.append(audio)
        
        return end_of_speech and len(self.audio_buffer) > 0
    
    async def process(self) -> np.ndarray | None:
        """Process accumulated audio and return TTS response"""
        if not self.audio_buffer or self.is_processing:
            return None
        
        self.is_processing = True
        
        try:
            # Concatenate audio
            audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            self.vad.reset()
            
            logger.info("🎤 Transcribing...")
            text = self.stt.transcribe(audio, config.sample_rate)
            
            if not text.strip():
                logger.info("(No speech detected)")
                return None
            
            logger.info(f"📝 User said: {text}")
            
            # Check for vision trigger
            if self._check_vision_trigger(text):
                logger.info("👁️ Vision request detected")
                image_base64 = self.vision.capture()
                
                if image_base64:
                    logger.info("🤖 Analyzing image...")
                    response = await self.llm.generate_with_vision(text, image_base64)
                else:
                    response = "Sorry, I couldn't access the camera to see anything."
            else:
                # Normal text response
                logger.info("🤖 Thinking...")
                response = await self.llm.generate(text)
            
            logger.info(f"💬 Response: {response}")
            
            # TTS
            logger.info("🔊 Generating speech...")
            audio_response = self.tts.synthesize(response)
            
            if audio_response is not None:
                self.tts_playing = True
                logger.info(f"✓ Generated {len(audio_response)/config.tts_sample_rate:.1f}s of audio")
            
            return audio_response
            
        finally:
            self.is_processing = False
    
    def cleanup(self):
        """Cleanup resources"""
        self.vision.release()


# ============================================================
# LiveKit Agent
# ============================================================

class VoiceAgent:
    """LiveKit Voice Agent with Vision"""
    
    def __init__(self, room_name: str = None):
        self.room_name = room_name or config.room_name
        self.room = rtc.Room()
        self.processor = AudioProcessor()
        self.audio_source = None
        self.running = False
    
    async def connect(self):
        """Connect to LiveKit room"""
        # Generate token
        token = api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
        token.with_identity(config.agent_identity)
        token.with_name(config.agent_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=self.room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        jwt_token = token.to_jwt()
        
        logger.info(f"🔗 Connecting to room: {self.room_name}")
        
        # Set up event handlers
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"👤 Participant joined: {participant.identity}")
            # Greet the new participant
            asyncio.create_task(self._say(config.greeting))
        
        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            logger.info(f"👤 Participant left: {participant.identity}")
        
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"🎵 Subscribed to audio from: {participant.identity}")
                asyncio.create_task(self._handle_audio_track(track, participant))
        
        @self.room.on("disconnected")
        def on_disconnected():
            logger.info("❌ Disconnected from room")
            self.running = False
        
        # Connect
        await self.room.connect(config.livekit_url, jwt_token)
        logger.info(f"✓ Connected to room: {self.room.name}")
        
        self.running = True
        
        # Create and publish audio source for TTS
        self.audio_source = rtc.AudioSource(config.sample_rate, config.channels)
        local_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self.audio_source)
        
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(local_track, options)
        logger.info("✓ Audio track published")
        
        # Print info
        logger.info("═" * 50)
        logger.info("✓ Voice Agent Active")
        logger.info(f"  Room: {self.room_name}")
        logger.info(f"  STT: Whisper ({config.whisper_model})")
        logger.info(f"  LLM: Ollama ({config.text_model})")
        logger.info(f"  Vision: {config.vision_model if config.vision_enabled else 'Disabled'}")
        logger.info(f"  TTS: Kokoro ({config.tts_voice})")
        logger.info("═" * 50)
        logger.info("Waiting for participants...")
        
        # Pre-load TTS to avoid delay on first message
        try:
            self.processor.tts.load()
        except Exception as e:
            logger.warning(f"Could not preload TTS: {e}")
    
    async def _handle_audio_track(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        """Handle incoming audio from participant"""
        audio_stream = rtc.AudioStream(track)
        
        async for event in audio_stream:
            if not self.running:
                break
            
            frame = event.frame
            
            # Convert to numpy
            audio_data = np.frombuffer(bytes(frame.data), dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to processor
            if self.processor.add_audio(audio_data, frame.sample_rate):
                # Process and respond
                response_audio = await self.processor.process()
                
                if response_audio is not None:
                    await self._send_audio(response_audio, config.tts_sample_rate)
                    self.processor.tts_playing = False
    
    async def _say(self, text: str):
        """Say something using TTS"""
        logger.info(f"🔊 Saying: {text}")
        audio = self.processor.tts.synthesize(text)
        if audio is not None:
            logger.info(f"📢 Sending {len(audio)/config.tts_sample_rate:.1f}s of audio...")
            self.processor.tts_playing = True
            await self._send_audio(audio, config.tts_sample_rate)
            self.processor.tts_playing = False
            logger.info("✓ Audio sent")
        else:
            logger.error("❌ TTS returned no audio")
    
    async def _send_audio(self, audio: np.ndarray, source_rate: int):
        """Send audio through LiveKit"""
        from scipy import signal
        
        # Convert to float for resampling
        audio_float = audio.astype(np.float32)
        
        # Resample to output sample rate if needed
        if source_rate != config.sample_rate:
            num_samples = int(len(audio_float) * config.sample_rate / source_rate)
            audio_float = signal.resample(audio_float, num_samples)
        
        # Convert back to int16 with proper clipping
        audio_int16 = np.clip(audio_float, -32768, 32767).astype(np.int16)
        
        # Send in chunks
        chunk_size = config.chunk_size
        total_chunks = (len(audio_int16) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(audio_int16), chunk_size):
            if not self.running:
                break
            
            chunk = audio_int16[i:i+chunk_size]
            
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Create frame and copy data
            frame = rtc.AudioFrame.create(config.sample_rate, config.channels, chunk_size)
            frame_data = np.frombuffer(frame.data, dtype=np.int16)
            np.copyto(frame_data, chunk)
            
            await self.audio_source.capture_frame(frame)
            
            # Wait for real-time playback (20ms per chunk at 48kHz)
            await asyncio.sleep(0.019)
    
    async def run(self):
        """Run the agent"""
        while self.running:
            await asyncio.sleep(0.3)
    
    async def disconnect(self):
        """Disconnect from room"""
        self.running = False
        self.processor.cleanup()
        await self.room.disconnect()
        logger.info("✓ Disconnected")


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="LiveKit AI Voice Agent with Vision")
    parser.add_argument("--room", default=None, help="Room name (overrides config)")
    parser.add_argument("--config", default="config.yml", help="Config file path")
    args = parser.parse_args()
    
    # Reload config if custom path specified
    global config
    if args.config != "config.yml":
        config = Config(args.config)
    
    print("═" * 50)
    print("  VChat - LiveKit AI Voice Agent")
    print("═" * 50)
    print(f"  Config: {args.config}")
    print(f"  Vision: {'Enabled' if config.vision_enabled else 'Disabled'}")
    print("═" * 50)
    
    agent = VoiceAgent(args.room)
    
    try:
        await agent.connect()
        await agent.run()
    except KeyboardInterrupt:
        print("\n⏹ Stopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
