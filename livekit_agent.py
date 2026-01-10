#!/usr/bin/env python3
"""
LiveKit AI Voice Agent
- Receives audio from LiveKit room participants
- Transcribes with Whisper STT (local)
- Generates response with Ollama LLM
- Sends TTS response back via LiveKit

Usage:
  python livekit_agent.py                     # Connect to room
  python livekit_agent.py --room my-room      # Specify room name
"""
import asyncio
import argparse
import os
import logging
import json
from collections import deque

import numpy as np
import torch
import aiohttp

from livekit import rtc, api

# Try to load dotenv, but don't fail if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vchat-agent")

# ============================================================
# Configuration
# ============================================================

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# AI Model Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

# Audio settings
SAMPLE_RATE = 48000
WHISPER_RATE = 16000
TTS_RATE = 24000


# ============================================================
# Whisper STT
# ============================================================

class WhisperSTT:
    """Local Whisper-based Speech-to-Text"""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
            language="en",
            fp16=self.device == "cuda",
            temperature=0,
            no_speech_threshold=0.5,
        )
        
        return result.get("text", "").strip()


# ============================================================
# Ollama LLM
# ============================================================

class OllamaLLM:
    """Ollama-based Language Model"""
    
    def __init__(self, model: str = "qwen2.5:0.5b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def generate(self, text: str) -> str:
        """Generate response from LLM"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": text,
                        "stream": False,
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "")
                    else:
                        logger.error(f"Ollama error: {resp.status}")
                        return "Sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "Sorry, I couldn't connect to the language model."


# ============================================================
# Kokoro TTS
# ============================================================

class KokoroTTS:
    """Kokoro Text-to-Speech"""
    
    def __init__(self, voice: str = "af_heart"):
        self.voice = voice
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
        
        results = self.pipeline(text, voice=self.voice, speed=1.0)
        
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
# Voice Activity Detection (Simple RMS-based)
# ============================================================

class SimpleVAD:
    """Simple Voice Activity Detection using RMS"""
    
    def __init__(self, threshold: float = 0.02, silence_duration: float = 1.0):
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.silence_samples = 0
        self.is_speaking = False
    
    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[bool, bool]:
        """
        Process audio chunk and return (is_speech, end_of_speech)
        """
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > self.threshold:
            self.is_speaking = True
            self.silence_samples = 0
            return True, False
        else:
            if self.is_speaking:
                self.silence_samples += len(audio)
                silence_time = self.silence_samples / sample_rate
                
                if silence_time >= self.silence_duration:
                    self.is_speaking = False
                    self.silence_samples = 0
                    return False, True  # End of speech
                
                return True, False  # Still in speech (brief pause)
            
            return False, False
    
    def reset(self):
        self.silence_samples = 0
        self.is_speaking = False


# ============================================================
# Audio Processor
# ============================================================

class AudioProcessor:
    """Processes audio: STT → LLM → TTS"""
    
    def __init__(self):
        self.stt = WhisperSTT(WHISPER_MODEL)
        self.llm = OllamaLLM(OLLAMA_MODEL, OLLAMA_BASE_URL)
        self.tts = KokoroTTS(KOKORO_VOICE)
        self.vad = SimpleVAD(threshold=0.02, silence_duration=1.0)
        
        self.audio_buffer = []
        self.is_processing = False
        self.tts_playing = False
    
    def add_audio(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Add audio chunk, return True if ready to process"""
        # Check for barge-in
        if self.tts_playing:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0.05:  # Strong speech = barge-in
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
            text = self.stt.transcribe(audio, SAMPLE_RATE)
            
            if not text.strip():
                logger.info("(No speech detected)")
                return None
            
            logger.info(f"📝 User said: {text}")
            
            # LLM
            logger.info("🤖 Thinking...")
            response = await self.llm.generate(text)
            logger.info(f"💬 Response: {response}")
            
            # TTS
            logger.info("🔊 Generating speech...")
            audio_response = self.tts.synthesize(response)
            
            if audio_response is not None:
                self.tts_playing = True
                logger.info(f"✓ Generated {len(audio_response)/TTS_RATE:.1f}s of audio")
            
            return audio_response
            
        finally:
            self.is_processing = False


# ============================================================
# LiveKit Agent
# ============================================================

class VoiceAgent:
    """LiveKit Voice Agent"""
    
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.room = rtc.Room()
        self.processor = AudioProcessor()
        self.audio_source = None
        self.running = False
    
    async def connect(self):
        """Connect to LiveKit room"""
        # Generate token
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity("ai-agent")
        token.with_name("AI Voice Agent")
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
        await self.room.connect(LIVEKIT_URL, jwt_token)
        logger.info(f"✓ Connected to room: {self.room.name}")
        
        self.running = True
        
        # Create and publish audio source for TTS
        self.audio_source = rtc.AudioSource(SAMPLE_RATE, 1)
        local_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self.audio_source)
        
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(local_track, options)
        logger.info("✓ Audio track published")
        
        # Print info
        logger.info("═" * 50)
        logger.info("✓ Voice Agent Active")
        logger.info(f"  Room: {self.room_name}")
        logger.info(f"  STT: Whisper ({WHISPER_MODEL})")
        logger.info(f"  LLM: Ollama ({OLLAMA_MODEL})")
        logger.info(f"  TTS: Kokoro ({KOKORO_VOICE})")
        logger.info("═" * 50)
        
        # Say hello
        await self._say("Hello! I'm your voice assistant. How can I help you today?")
    
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
                    await self._send_audio(response_audio, TTS_RATE)
                    self.processor.tts_playing = False
    
    async def _say(self, text: str):
        """Say something using TTS"""
        logger.info(f"🔊 Saying: {text}")
        audio = self.processor.tts.synthesize(text)
        if audio is not None:
            self.processor.tts_playing = True
            await self._send_audio(audio, TTS_RATE)
            self.processor.tts_playing = False
    
    async def _send_audio(self, audio: np.ndarray, source_rate: int):
        """Send audio through LiveKit"""
        # Resample to 48kHz if needed
        if source_rate != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(audio) * SAMPLE_RATE / source_rate)
            audio = signal.resample(audio, num_samples).astype(np.int16)
        
        # Send in chunks
        chunk_size = 960  # 20ms at 48kHz
        
        for i in range(0, len(audio), chunk_size):
            if not self.running:
                break
            
            chunk = audio[i:i+chunk_size]
            
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Create frame
            frame = rtc.AudioFrame.create(SAMPLE_RATE, 1, chunk_size)
            np.copyto(np.frombuffer(frame.data, dtype=np.int16), chunk)
            
            await self.audio_source.capture_frame(frame)
            await asyncio.sleep(0.018)  # Slightly less than 20ms to prevent underruns
    
    async def run(self):
        """Run the agent"""
        while self.running:
            await asyncio.sleep(1)
    
    async def disconnect(self):
        """Disconnect from room"""
        self.running = False
        await self.room.disconnect()
        logger.info("✓ Disconnected")


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="LiveKit AI Voice Agent")
    parser.add_argument("--room", default="vchat-room", help="Room name")
    args = parser.parse_args()
    
    print("═" * 50)
    print("  VChat - LiveKit AI Voice Agent")
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
