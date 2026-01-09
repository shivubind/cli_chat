#!/usr/bin/env python3
"""
Smart Audio Server with STT + LLM + TTS
- Receives audio from WebRTC client
- Transcribes with Whisper STT
- Generates response with Ollama LLM
- Sends TTS response audio back to client via WebRTC
"""
import asyncio
import sys
import aiohttp
import pyaudio
import numpy as np
import whisper
import torch
import fractions
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack
from av import AudioFrame

SIGNALING_SERVER = "http://127.0.0.1:8080"
PEER_ID = "audio_call"

# Audio settings
SAMPLE_RATE = 48000
WHISPER_RATE = 16000  # Whisper expects 16kHz
TTS_RATE = 24000  # Kokoro outputs 24kHz

# STT/LLM/TTS Config
WHISPER_MODEL = "small"  # base is more accurate than tiny (tiny→base→small→medium→large)
OLLAMA_MODEL = "qwen2.5:0.5b"  # Change to your model
KOKORO_VOICE = "af_heart"


class WhisperSTT:
    """Whisper-based Speech-to-Text"""
    def __init__(self, model_name="base"):
        print(f"Loading Whisper model: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=device)
        print(f"✓ Whisper loaded on {device}")
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text"""
        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to [-1, 1] range
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Pad or trim to 30 seconds (Whisper expects this)
        # This helps with short audio clips
        target_length = 16000 * 30  # 30 seconds at 16kHz
        if len(audio_data) < target_length:
            # Pad with silence
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        
        result = self.model.transcribe(
            audio_data, 
            language="en",
            fp16=torch.cuda.is_available(),  # Use fp16 on GPU for speed
            temperature=0,  # More deterministic
            no_speech_threshold=0.5,  # Filter out non-speech
            condition_on_previous_text=False  # Each transcription is independent
        )
        return result['text'].strip()


class OllamaLLM:
    """Ollama-based Language Model"""
    def __init__(self, model_name="qwen2.5:0.5b"):
        self.model = model_name
        self.base_url = "http://localhost:11434"
    
    async def generate(self, text: str) -> str:
        """Generate response from LLM"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": text,
                        "stream": False
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('response', '')
                    else:
                        return "Sorry, I couldn't generate a response."
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Sorry, I couldn't connect to the language model."


class KokoroTTS:
    """Kokoro Text-to-Speech - returns audio numpy array"""
    def __init__(self, voice="af_heart"):
        self.voice = voice
        self.pipeline = None
        self._init_kokoro()
    
    def _init_kokoro(self):
        try:
            from kokoro import KPipeline
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.pipeline = KPipeline(
                lang_code='en-us',
                repo_id='hexgrad/Kokoro-82M',
                device=device
            )
            print(f"✓ Kokoro TTS loaded on {device}")
        except ImportError:
            print("⚠ Kokoro not installed")
            self.pipeline = None
        except Exception as e:
            print(f"⚠ Kokoro failed: {e}")
            self.pipeline = None
    
    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio numpy array (24kHz, int16)"""
        if not self.pipeline:
            return None
        
        # Generate audio
        results = self.pipeline(text, voice=self.voice, speed=1.0)
        
        audio_chunks = []
        for result in results:
            if result.audio is not None:
                audio_chunks.append(result.audio)
        
        if not audio_chunks:
            return None
        
        # Concatenate
        audio = torch.cat(audio_chunks, dim=-1)
        audio_np = audio.cpu().numpy()
        audio_np = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        
        return audio_np


class TTSAudioTrack(MediaStreamTrack):
    """WebRTC audio track that sends TTS audio"""
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._timestamp = 0
        self._audio_queue = asyncio.Queue()
        self._current_audio = None
        self._current_pos = 0
        self._sample_rate = SAMPLE_RATE
        self._chunk_size = 960  # 20ms at 48kHz
        self.is_speaking = False  # Flag to track if TTS is playing
    
    def queue_audio(self, audio_np: np.ndarray, source_rate: int):
        """Queue audio for sending (will resample if needed)"""
        from scipy import signal
        
        # Resample to 48kHz if needed
        if source_rate != self._sample_rate:
            num_samples = int(len(audio_np) * self._sample_rate / source_rate)
            audio_np = signal.resample(audio_np, num_samples).astype(np.int16)
        
        self._audio_queue.put_nowait(audio_np)
        self.is_speaking = True
    
    def stop_speaking(self):
        """Stop current TTS playback (barge-in)"""
        if self.is_speaking:
            print("🛑 Interrupted! Stopping TTS...")
            self._current_audio = None
            self._current_pos = 0
            # Clear queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except:
                    break
            self.is_speaking = False
    
    async def recv(self):
        """Send audio frame to client"""
        # Small delay to not block event loop (20ms = real-time audio)
        await asyncio.sleep(0.02)
        
        # Get audio data for this frame
        audio_chunk = np.zeros(self._chunk_size, dtype=np.int16)
        
        # Fill from current audio or get new audio
        if self._current_audio is not None and self._current_pos < len(self._current_audio):
            # Continue sending current audio
            end_pos = min(self._current_pos + self._chunk_size, len(self._current_audio))
            chunk_len = end_pos - self._current_pos
            audio_chunk[:chunk_len] = self._current_audio[self._current_pos:end_pos]
            self._current_pos = end_pos
        else:
            # Finished current audio
            if self._current_audio is not None:
                self._current_audio = None
            
            # Check for new audio in queue
            try:
                self._current_audio = self._audio_queue.get_nowait()
                self._current_pos = 0
                
                # Send first chunk
                end_pos = min(self._chunk_size, len(self._current_audio))
                audio_chunk[:end_pos] = self._current_audio[:end_pos]
                self._current_pos = end_pos
            except asyncio.QueueEmpty:
                # No more audio to send
                self.is_speaking = False
        
        # Create audio frame
        audio_array = audio_chunk.reshape(1, -1)
        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        
        self._timestamp += self._chunk_size
        
        return frame


class AudioProcessor:
    """Processes incoming audio: STT → LLM → TTS → Send back"""
    def __init__(self, tts_track: TTSAudioTrack):
        self.stt = WhisperSTT(WHISPER_MODEL)
        self.llm = OllamaLLM(OLLAMA_MODEL)
        self.tts = KokoroTTS(KOKORO_VOICE)
        self.tts_track = tts_track
        
        self.audio_buffer = []
        self.buffer_duration = 0
        self.min_audio_duration = 1.5  # Reduced for faster response
        
        # Better VAD thresholds
        self.silence_threshold = 0.02  # Higher threshold to ignore background
        self.speech_threshold = 0.05   # Strong speech for barge-in detection
        self.silence_duration = 0
        self.max_silence = 1.0  # Faster end-of-speech detection
        
        self.is_processing = False  # Flag to prevent re-entry
    
    def add_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Add audio chunk to buffer with better VAD"""
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        # BARGE-IN: If TTS is speaking and strong speech detected, interrupt
        if self.tts_track.is_speaking:
            if rms > self.speech_threshold:
                self.tts_track.stop_speaking()
                # Start capturing the new speech
                self.audio_buffer = [audio_data]
                self.buffer_duration = len(audio_data) / sample_rate
                self.silence_duration = 0
            # Ignore audio while TTS is speaking (unless barge-in)
            return
        
        # If processing, ignore new audio
        if self.is_processing:
            return
        
        # Normal VAD
        if rms > self.silence_threshold:
            self.silence_duration = 0
            self.audio_buffer.append(audio_data)
            self.buffer_duration += len(audio_data) / sample_rate
        else:
            self.silence_duration += len(audio_data) / sample_rate
            if self.audio_buffer:
                self.audio_buffer.append(audio_data)
                self.buffer_duration += len(audio_data) / sample_rate
    
    def should_process(self) -> bool:
        """Check if we should process the buffer"""
        return (
            not self.is_processing and
            not self.tts_track.is_speaking and
            self.buffer_duration >= self.min_audio_duration and
            self.silence_duration >= self.max_silence
        )
    
    async def process(self):
        """Process accumulated audio and send response"""
        if not self.audio_buffer or self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            # Concatenate audio
            audio = np.concatenate(self.audio_buffer)
            
            # Resample to 16kHz for Whisper
            from scipy import signal
            audio_16k = signal.resample(audio, int(len(audio) * WHISPER_RATE / SAMPLE_RATE))
            
            if len(audio_16k.shape) > 1:
                audio_16k = audio_16k.mean(axis=1)
            
            print("\n" + "=" * 50)
            print("🎤 Transcribing...")
            
            # STT
            text = self.stt.transcribe(audio_16k)
            
            if text.strip():
                print(f"📝 YOU SAID: {text}")
                
                # LLM
                print("🤖 Thinking...")
                response = await self.llm.generate(text)
                
                print(f"💬 RESPONSE: {response}")
                
                # TTS - generate audio and send to client
                print("🔊 Sending audio response to client...")
                audio_np = self.tts.synthesize(response)
                
                if audio_np is not None:
                    self.tts_track.queue_audio(audio_np, TTS_RATE)
                    print(f"✓ Sent {len(audio_np)/TTS_RATE:.1f}s of audio")
                else:
                    print("⚠ TTS failed")
            else:
                print("(No speech detected)")
            
            print("=" * 50 + "\n")
        
        finally:
            # Always clear buffer and reset state
            self.audio_buffer = []
            self.buffer_duration = 0
            self.silence_duration = 0
            self.is_processing = False


class SmartAudioReceiver:
    """Receives audio and processes with AI (no local playback)"""
    def __init__(self, processor: AudioProcessor):
        self.running = True
        self.processor = processor
    
    async def receive_and_process(self, track):
        """Receive audio and process (no local playback)"""
        print("▶ Receiving audio stream...")
        
        try:
            while self.running:
                frame = await track.recv()
                
                # Get audio bytes
                if frame.format.name == 's16':
                    audio_bytes = bytes(frame.planes[0])
                else:
                    arr = frame.to_ndarray().flatten()
                    audio_bytes = (np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes()
                
                # NOTE: Not playing locally - just processing
                
                # Convert to numpy for processing
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Add to processor
                self.processor.add_audio(audio_np, frame.sample_rate)
                
                # Check if should process
                if self.processor.should_process():
                    await self.processor.process()
                    
        except Exception as e:
            if "MediaStreamError" not in str(type(e).__name__):
                print(f"Error: {e}")
    
    def stop(self):
        self.running = False


async def run_smart_server():
    """Run as WebRTC receiver with AI processing"""
    config = RTCConfiguration(iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"])
    ])
    
    pc = RTCPeerConnection(configuration=config)
    
    # Create TTS audio track for sending responses
    tts_track = TTSAudioTrack()
    pc.addTrack(tts_track)
    print("✓ TTS audio track ready (will send responses to client)")
    
    # Create processor with TTS track
    processor = AudioProcessor(tts_track)
    receiver = SmartAudioReceiver(processor)
    receive_task = None
    
    @pc.on("connectionstatechange")
    async def on_state():
        print(f"Connection: {pc.connectionState}")
        if pc.connectionState == "connected":
            print("═" * 50)
            print("✓ CONNECTED! Smart Server Active")
            print("  - Receiving audio from client")
            print("  - STT: Whisper")
            print("  - LLM: Ollama")
            print("  - TTS: Kokoro → Sent to client")
            print("═" * 50)
    
    @pc.on("track")
    async def on_track(track):
        nonlocal receive_task
        if track.kind == "audio":
            print("✓ Client audio track received")
            receive_task = asyncio.create_task(receiver.receive_and_process(track))
    
    async with aiohttp.ClientSession() as session:
        # Wait for caller's offer
        print("Waiting for caller...")
        while True:
            async with session.get(f"{SIGNALING_SERVER}/offer?peer_id={PEER_ID}") as r:
                data = await r.json()
                if data['sdp']:
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data['sdp']['sdp'], type=data['sdp']['type'])
                    )
                    break
            await asyncio.sleep(0.1)
        
        # Create and send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Wait for ICE gathering (with timeout)
        for _ in range(50):  # 5 second timeout
            if pc.iceGatheringState == "complete":
                break
            await asyncio.sleep(0.1)
        print(f"ICE state: {pc.iceGatheringState}")
        
        await session.post(f"{SIGNALING_SERVER}/answer", json={
            'peer_id': PEER_ID,
            'sdp': {'type': pc.localDescription.type, 'sdp': pc.localDescription.sdp}
        })
        print("Answer sent")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        receiver.stop()
        if receive_task:
            receive_task.cancel()
        await pc.close()


if __name__ == "__main__":
    print("=" * 50)
    print("Smart Audio Server")
    print("STT: Whisper | LLM: Ollama | TTS: Kokoro")
    print("Response audio sent to client via WebRTC")
    print("=" * 50)
    asyncio.run(run_smart_server())
