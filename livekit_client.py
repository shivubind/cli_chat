#!/usr/bin/env python3
"""
LiveKit Audio/Video Client with Vision Support
- Connects to a LiveKit room
- Captures microphone audio and publishes to room
- Captures camera video and publishes to room (for AI vision)
- Plays back audio from other participants (AI agent)

Usage:
  python livekit_client.py                    # Uses env vars for connection
  python livekit_client.py --room my-room     # Specify room name
  python livekit_client.py --no-video         # Audio only, no camera
"""
import asyncio
import argparse
import os
import sys
import threading
from dotenv import load_dotenv

import numpy as np
import pyaudio
from livekit import rtc, api

# Load environment variables
load_dotenv()

# ============================================================
# Configuration
# ============================================================

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# Audio settings
SAMPLE_RATE = 48000
CHANNELS = 1
BUFFER_SIZE = 960  # 20ms at 48kHz
PLAYBACK_BUFFER = 4800  # 100ms buffer

# Video settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 15  # Lower FPS to reduce bandwidth


# ============================================================
# Audio Source (Microphone)
# ============================================================

class MicrophoneSource:
    """Captures audio from microphone using PyAudio"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        
    def start(self):
        """Start audio capture"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=BUFFER_SIZE,
        )
        self.running = True
        print(f"🎤 Microphone started ({self.sample_rate}Hz, {self.channels}ch)")
    
    def read(self) -> bytes:
        """Read audio chunk from microphone"""
        if not self.stream:
            return b''
        return self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
    
    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("🎤 Microphone stopped")


# ============================================================
# Audio Playback (Speaker)
# ============================================================

class AudioPlayback:
    """Plays audio to speakers using PyAudio"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.sample_rate = None
        self.channels = None
        
    def init_stream(self, sample_rate: int, channels: int):
        """Initialize or reinitialize playback stream"""
        if self.stream and self.sample_rate == sample_rate and self.channels == channels:
            return
        
        if self.stream:
            self.stream.close()
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=PLAYBACK_BUFFER,
        )
        print(f"🔊 Speaker initialized ({sample_rate}Hz, {channels}ch)")
    
    def play(self, audio_data: bytes):
        """Play audio bytes"""
        if self.stream:
            self.stream.write(audio_data)
    
    def stop(self):
        """Stop playback"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("🔊 Speaker stopped")


# ============================================================
# Video Source (Camera)
# ============================================================

class CameraSource:
    """Captures video from camera using OpenCV"""
    
    def __init__(self, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT, 
                 fps: int = VIDEO_FPS, camera_index: int = 0):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.camera = None
        self.running = False
        self.cv2 = None
        self.latest_frame = None
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start video capture"""
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            print("❌ OpenCV not installed. Run: pip install opencv-python")
            return False
        
        print(f"📷 Opening camera {self.camera_index}...")
        self.camera = self.cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            print(f"❌ Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.camera.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(self.cv2.CAP_PROP_FPS, self.fps)
        
        # Warm up camera
        for _ in range(5):
            self.camera.read()
        
        # Get actual resolution
        actual_w = int(self.camera.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = actual_w
        self.height = actual_h
        
        self.running = True
        print(f"📷 Camera started ({actual_w}x{actual_h} @ {self.fps}fps)")
        return True
    
    def read(self) -> np.ndarray | None:
        """Read a frame from camera (BGR format)"""
        if not self.camera or not self.running:
            return None
        
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None
        
        with self._lock:
            self.latest_frame = frame.copy()
        
        return frame
    
    def get_latest_frame(self) -> np.ndarray | None:
        """Get the most recently captured frame"""
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop(self):
        """Stop video capture"""
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print("📷 Camera stopped")


# ============================================================
# LiveKit Client with Video
# ============================================================

class LiveKitClient:
    """LiveKit room client with microphone, speaker, and camera"""
    
    def __init__(self, url: str, token: str, room_name: str, enable_video: bool = True):
        self.url = url
        self.token = token
        self.room_name = room_name
        self.enable_video = enable_video
        
        self.room = rtc.Room()
        self.mic = MicrophoneSource()
        self.speaker = AudioPlayback()
        self.camera = CameraSource() if enable_video else None
        
        self.audio_source = None
        self.video_source = None
        self.local_audio_track = None
        self.local_video_track = None
        
        self.running = False
        
    async def connect(self):
        """Connect to LiveKit room"""
        print(f"🔗 Connecting to room: {self.room_name}")
        print(f"   URL: {self.url}")
        
        # Set up event handlers
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            print(f"👤 Participant joined: {participant.identity}")
        
        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            print(f"👤 Participant left: {participant.identity}")
        
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                print(f"🎵 Subscribed to audio from: {participant.identity}")
                asyncio.create_task(self._handle_audio_track(track))
        
        @self.room.on("disconnected")
        def on_disconnected():
            print("❌ Disconnected from room")
            self.running = False
        
        @self.room.on("connection_state_changed")
        def on_connection_state(state: rtc.ConnectionState):
            print(f"📡 Connection state: {state}")
        
        # Connect to room
        await self.room.connect(self.url, self.token)
        print(f"✓ Connected to room: {self.room.name}")
        
        # Print existing participants
        for participant in self.room.remote_participants.values():
            print(f"👤 Existing participant: {participant.identity}")
        
        self.running = True
    
    async def _handle_audio_track(self, track: rtc.Track):
        """Handle incoming audio from remote participant"""
        audio_stream = rtc.AudioStream(track)
        frame_count = 0
        
        async for event in audio_stream:
            if not self.running:
                break
            
            frame = event.frame
            
            # Initialize speaker if needed
            self.speaker.init_stream(frame.sample_rate, frame.num_channels)
            
            # Get audio data and play
            audio_data = bytes(frame.data)
            
            # Check if there's actual audio (not silence)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            if rms > 100:  # Non-silence threshold
                if frame_count == 0:
                    print(f"🔊 Receiving audio... (rms={rms:.0f})")
                frame_count += 1
            
            self.speaker.play(audio_data)
    
    async def publish_microphone(self):
        """Publish microphone audio to room"""
        # Create audio source
        self.audio_source = rtc.AudioSource(SAMPLE_RATE, CHANNELS)
        
        # Create local audio track
        self.local_audio_track = rtc.LocalAudioTrack.create_audio_track(
            "microphone", 
            self.audio_source
        )
        
        # Publish track
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE,
        )
        publication = await self.room.local_participant.publish_track(
            self.local_audio_track, 
            options
        )
        print(f"✓ Microphone published: {publication.sid}")
        
        # Start microphone capture
        self.mic.start()
        
        # Capture and send audio frames
        while self.running:
            try:
                audio_bytes = self.mic.read()
                if audio_bytes:
                    # Create audio frame
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    frame = rtc.AudioFrame.create(
                        SAMPLE_RATE,
                        CHANNELS,
                        len(audio_data),
                    )
                    # Copy audio data to frame
                    np.copyto(
                        np.frombuffer(frame.data, dtype=np.int16),
                        audio_data
                    )
                    
                    # Capture frame to source
                    await self.audio_source.capture_frame(frame)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self.running:
                    print(f"Error capturing audio: {e}")
                await asyncio.sleep(0.01)
            
            # Small yield to prevent blocking
            await asyncio.sleep(0.001)
    
    async def publish_camera(self):
        """Publish camera video to room"""
        if not self.enable_video or not self.camera:
            return
        
        # Start camera
        if not self.camera.start():
            print("⚠️ Camera not available, continuing without video")
            self.enable_video = False
            return
        
        # Create video source (RGBA format for LiveKit)
        self.video_source = rtc.VideoSource(self.camera.width, self.camera.height)
        
        # Create local video track
        self.local_video_track = rtc.LocalVideoTrack.create_video_track(
            "camera",
            self.video_source
        )
        
        # Publish track
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
        )
        publication = await self.room.local_participant.publish_track(
            self.local_video_track,
            options
        )
        print(f"✓ Camera published: {publication.sid}")
        
        # Frame interval for target FPS
        frame_interval = 1.0 / VIDEO_FPS
        
        # Capture and send video frames
        while self.running:
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Read frame from camera
                frame_bgr = self.camera.read()
                
                if frame_bgr is not None:
                    # Convert BGR to RGBA for LiveKit
                    frame_rgba = self.camera.cv2.cvtColor(frame_bgr, self.camera.cv2.COLOR_BGR2RGBA)
                    
                    # Create VideoFrame
                    video_frame = rtc.VideoFrame(
                        self.camera.width,
                        self.camera.height,
                        rtc.VideoBufferType.RGBA,
                        frame_rgba.tobytes()
                    )
                    
                    # Capture frame
                    self.video_source.capture_frame(video_frame)
                
                # Maintain target FPS
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error capturing video: {e}")
                await asyncio.sleep(0.1)
    
    async def disconnect(self):
        """Disconnect from room and cleanup"""
        self.running = False
        self.mic.stop()
        self.speaker.stop()
        if self.camera:
            self.camera.stop()
        await self.room.disconnect()
        print("✓ Disconnected from room")


# ============================================================
# Token Generation
# ============================================================

def create_token(room_name: str, identity: str) -> str:
    """Create access token for room with clock skew compensation"""
    import json
    import base64
    import hmac
    import hashlib
    
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_name(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_sources=["camera", "microphone"],
    ))
    
    # Generate initial JWT
    jwt_token = token.to_jwt()
    
    # Decode and modify nbf to subtract 10 seconds for clock skew compensation
    try:
        # Decode without verification to modify the claims
        parts = jwt_token.split('.')
        payload = parts[1]
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding and padding != 4:
            payload += '=' * padding
        
        decoded_payload = json.loads(base64.urlsafe_b64decode(payload))
        
        # Subtract 10 seconds from nbf to account for clock skew
        if 'nbf' in decoded_payload:
            decoded_payload['nbf'] -= 10
        
        # Re-encode
        modified_payload = base64.urlsafe_b64encode(
            json.dumps(decoded_payload, separators=(',', ':')).encode()
        ).decode().rstrip('=')
        
        # Re-sign with the secret using HMAC-SHA256
        header_part = parts[0]
        message = f"{header_part}.{modified_payload}"
        signature = base64.urlsafe_b64encode(
            hmac.new(
                LIVEKIT_API_SECRET.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode().rstrip('=')
        
        jwt_token = f"{message}.{signature}"
    except Exception as e:
        print(f"Warning: Failed to adjust token nbf: {e}, using original token")
    
    return jwt_token


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="LiveKit Audio/Video Client")
    parser.add_argument("--room", default="vchat-room", help="Room name")
    parser.add_argument("--url", default=LIVEKIT_URL, help="LiveKit server URL")
    parser.add_argument("--identity", default="user", help="Your identity")
    parser.add_argument("--no-video", action="store_true", help="Disable camera/video")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()
    
    enable_video = not args.no_video
    
    print("═" * 50)
    print("  VChat - LiveKit Audio/Video Client")
    print("═" * 50)
    print(f"  Video: {'Enabled' if enable_video else 'Disabled'}")
    print("═" * 50)
    
    # Generate token
    token = create_token(args.room, args.identity)
    
    # Create and connect client
    client = LiveKitClient(args.url, token, args.room, enable_video=enable_video)
    
    # Set camera index if video enabled
    if enable_video and client.camera:
        client.camera.camera_index = args.camera
    
    try:
        await client.connect()
        
        print("\n" + "═" * 50)
        print("✓ CONNECTED! Voice chat active")
        if enable_video:
            print("  📷 Camera is streaming to AI")
            print("  Say 'what do you see' for AI to describe your view")
        print("  🎤 Speak into your microphone...")
        print("  Press Ctrl+C to exit")
        print("═" * 50 + "\n")
        
        # Start audio and video publishing concurrently
        tasks = [asyncio.create_task(client.publish_microphone())]
        if enable_video:
            tasks.append(asyncio.create_task(client.publish_camera()))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        print("\n⏹ Stopping...")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()
        print("✓ Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
