#!/usr/bin/env python3
"""
WebRTC Audio Caller - LOW LATENCY
Run as: python3 caller.py [signaling_server_url]

Examples:
  python3 caller.py                           # Local: http://127.0.0.1:8080
  python3 caller.py http://192.168.1.100:8080 # Remote server
  python3 caller.py http://myserver.com:8080  # Public server
"""
import asyncio
import sys
import aiohttp
import pyaudio
import fractions
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack
from av import AudioFrame
import numpy as np

# Default signaling server (can be overridden via command line)
SIGNALING_SERVER = "http://127.0.0.1:8080"
PEER_ID = "audio_call"

# Audio settings
SAMPLE_RATE = 48000
CHANNELS = 1
BUFFER_SIZE = 960  # 20ms at 48kHz (for mic capture)
PLAYBACK_BUFFER = 4800  # 100ms buffer for network playback (prevents underruns)


class PyAudioMicTrack(MediaStreamTrack):
    """Audio track from microphone using PyAudio"""
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._timestamp = 0
        self._start()
    
    def _start(self):
        """Start audio capture"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=BUFFER_SIZE
        )
    
    async def recv(self):
        """Receive audio frame from microphone"""
        data = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
        audio_array = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
        
        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        
        self._timestamp += BUFFER_SIZE
        return frame
    
    def stop(self):
        """Stop audio capture"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class AudioPlaybackTrack:
    """Low-latency audio playback"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = True
    
    def init_stream(self, sample_rate, channels):
        if self.stream:
            self.stream.close()
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=PLAYBACK_BUFFER
        )
        print(f"▶ Audio: {sample_rate}Hz, {channels}ch, buffer={PLAYBACK_BUFFER}")
    
    async def play_track(self, track):
        """Play audio frames immediately"""
        print("▶ Playing audio...")
        initialized = False
        try:
            while self.running:
                frame = await track.recv()
                
                if not initialized:
                    self.init_stream(frame.sample_rate, len(frame.layout.channels))
                    initialized = True
                
                if frame.format.name == 's16':
                    audio_bytes = bytes(frame.planes[0])
                else:
                    arr = frame.to_ndarray().flatten()
                    audio_bytes = (np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes()
                
                if self.stream:
                    self.stream.write(audio_bytes)
                    
        except Exception as e:
            if "MediaStreamError" not in str(type(e).__name__):
                print(f"Error: {e}")
    
    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
        except:
            pass


async def run_caller(server_url):
    """Run as caller - initiates the call"""
    print("═" * 40)
    print("  CALLER - Initiating call...")
    print("═" * 40)
    
    # STUN for faster ICE
    config = RTCConfiguration(iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"])
    ])
    
    pc = RTCPeerConnection(configuration=config)
    player_track = AudioPlaybackTrack()
    playback_task = None
    mic_track = None
    has_mic = False
    
    # Setup microphone
    try:
        mic_track = PyAudioMicTrack()
        pc.addTrack(mic_track)
        has_mic = True
        print("✓ Mic ready (PyAudio)")
    except Exception as e:
        print(f"⚠ Mic error: {e}")
    
    # If no mic, create data channel so WebRTC can still connect
    if not has_mic:
        pc.createDataChannel("dummy")
        print("(Running in receive-only mode)")
    
    @pc.on("connectionstatechange")
    async def on_state():
        print(f"State: {pc.connectionState}")
        if pc.connectionState == "connected":
            print("═" * 40)
            print("✓ CONNECTED! Low latency mode")
            print("═" * 40)
    
    @pc.on("track")
    async def on_track(track):
        nonlocal playback_task
        if track.kind == "audio":
            print("✓ Audio track received from receiver")
            playback_task = asyncio.create_task(player_track.play_track(track))
    
    async with aiohttp.ClientSession() as session:
        # Create and send offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        # Wait for ICE gathering
        print("Gathering ICE candidates...")
        for _ in range(50):  # 5 second timeout
            if pc.iceGatheringState == "complete":
                break
            await asyncio.sleep(0.1)
        print(f"ICE state: {pc.iceGatheringState}")
        
        # Post offer to signaling server
        await session.post(f"{server_url}/offer", json={
            'peer_id': PEER_ID,
            'sdp': {'type': pc.localDescription.type, 'sdp': pc.localDescription.sdp}
        })
        print("✓ Offer sent to signaling server")
        print("Waiting for receiver to answer...")
        
        # Poll for answer
        while True:
            async with session.get(f"{server_url}/answer?peer_id={PEER_ID}") as r:
                data = await r.json()
                if data['sdp']:
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data['sdp']['sdp'], type=data['sdp']['type'])
                    )
                    print("✓ Answer received!")
                    break
            await asyncio.sleep(0.1)
    
    # Keep running
    try:
        print("\nCall active. Press Ctrl+C to end.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nEnding call...")
    finally:
        player_track.stop()
        if mic_track:
            mic_track.stop()
        if playback_task:
            playback_task.cancel()
        await pc.close()
        print("Call ended.")


if __name__ == "__main__":
    server = SIGNALING_SERVER
    if len(sys.argv) > 1:
        server = sys.argv[1]
    
    print(f"Signaling server: {server}")
    asyncio.run(run_caller(server))

