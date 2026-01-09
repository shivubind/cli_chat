#!/usr/bin/env python3
"""
Simple WebRTC Signaling Server - Exchanges SDP between peers
"""
import asyncio
import json
from aiohttp import web

# Store connected peers
peers = {}
offers = {}
answers = {}

async def handle_offer(request):
    """Receive offer from sender"""
    data = await request.json()
    peer_id = data.get('peer_id', 'default')
    offers[peer_id] = data['sdp']
    print(f"Received offer from {peer_id}")
    return web.json_response({'status': 'ok'})

async def get_offer(request):
    """Get offer for receiver"""
    peer_id = request.query.get('peer_id', 'default')
    if peer_id in offers:
        return web.json_response({'sdp': offers[peer_id]})
    return web.json_response({'sdp': None})

async def handle_answer(request):
    """Receive answer from receiver"""
    data = await request.json()
    peer_id = data.get('peer_id', 'default')
    answers[peer_id] = data['sdp']
    print(f"Received answer from {peer_id}")
    return web.json_response({'status': 'ok'})

async def get_answer(request):
    """Get answer for sender"""
    peer_id = request.query.get('peer_id', 'default')
    if peer_id in answers:
        return web.json_response({'sdp': answers[peer_id]})
    return web.json_response({'sdp': None})

async def handle_ice(request):
    """Handle ICE candidates"""
    data = await request.json()
    peer_id = data.get('peer_id', 'default')
    role = data.get('role', 'sender')
    
    key = f"{peer_id}_{role}"
    if key not in peers:
        peers[key] = []
    peers[key].append(data['candidate'])
    return web.json_response({'status': 'ok'})

async def get_ice(request):
    """Get ICE candidates"""
    peer_id = request.query.get('peer_id', 'default')
    role = request.query.get('role', 'receiver')
    
    # Get candidates from opposite role
    opposite = 'sender' if role == 'receiver' else 'receiver'
    key = f"{peer_id}_{opposite}"
    
    candidates = peers.get(key, [])
    return web.json_response({'candidates': candidates})

app = web.Application()
app.router.add_post('/offer', handle_offer)
app.router.add_get('/offer', get_offer)
app.router.add_post('/answer', handle_answer)
app.router.add_get('/answer', get_answer)
app.router.add_post('/ice', handle_ice)
app.router.add_get('/ice', get_ice)

if __name__ == '__main__':
    print("Signaling server running on http://0.0.0.0:8080")
    web.run_app(app, host='0.0.0.0', port=8080)

