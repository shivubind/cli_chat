#!/usr/bin/env python3
"""
LiveKit Audio Client
- Connects to a LiveKit room
- Captures microphone audio and publishes to room
- Plays back audio from other participants (AI agent)

Usage:
  python livekit_client.py                    # Uses env vars for connection
  python livekit_client.py --room my-room     # Specify room name
  python livekit_client.py --url wss://...    # Specify LiveKit server URL
"""
import asyncio
import argparse
import os
import sys
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
# LiveKit Client
# ============================================================

class LiveKitClient:
    """LiveKit room client with microphone and speaker"""
    
    def __init__(self, url: str, token: str, room_name: str):
        self.url = url
        self.token = token
        self.room_name = room_name
        self.room = rtc.Room()
        self.mic = MicrophoneSource()
        self.speaker = AudioPlayback()
        self.audio_source = None
        self.local_track = None
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
        
        async for event in audio_stream:
            if not self.running:
                break
            
            frame = event.frame
            
            # Initialize speaker if needed
            self.speaker.init_stream(frame.sample_rate, frame.num_channels)
            
            # Get audio data and play
            audio_data = bytes(frame.data)
            self.speaker.play(audio_data)
    
    async def publish_microphone(self):
        """Publish microphone audio to room"""
        # Create audio source
        self.audio_source = rtc.AudioSource(SAMPLE_RATE, CHANNELS)
        
        # Create local audio track
        self.local_track = rtc.LocalAudioTrack.create_audio_track(
            "microphone", 
            self.audio_source
        )
        
        # Publish track
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE,
        )
        publication = await self.room.local_participant.publish_track(
            self.local_track, 
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
                    
            except Exception as e:
                print(f"Error capturing audio: {e}")
                await asyncio.sleep(0.01)
            
            # Small yield to prevent blocking
            await asyncio.sleep(0.001)
    
    async def disconnect(self):
        """Disconnect from room and cleanup"""
        self.running = False
        self.mic.stop()
        self.speaker.stop()
        await self.room.disconnect()
        print("✓ Disconnected from room")


# ============================================================
# Token Generation
# ============================================================

def create_token(room_name: str, identity: str) -> str:
    """Create access token for room"""
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_name(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
    ))
    return token.to_jwt()


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="LiveKit Audio Client")
    parser.add_argument("--room", default="vchat-room", help="Room name")
    parser.add_argument("--url", default=LIVEKIT_URL, help="LiveKit server URL")
    parser.add_argument("--identity", default="user", help="Your identity")
    args = parser.parse_args()
    
    print("═" * 50)
    print("  VChat - LiveKit Audio Client")
    print("═" * 50)
    
    # Generate token
    token = create_token(args.room, args.identity)
    
    # Create and connect client
    client = LiveKitClient(args.url, token, args.room)
    
    try:
        await client.connect()
        
        print("\n" + "═" * 50)
        print("✓ CONNECTED! Voice chat active")
        print("  Speak into your microphone...")
        print("  Press Ctrl+C to exit")
        print("═" * 50 + "\n")
        
        # Publish microphone and keep running
        await client.publish_microphone()
        
    except KeyboardInterrupt:
        print("\n⏹ Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

