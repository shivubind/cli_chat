#!/usr/bin/env python3
"""
Test script for agent tools
Run this to verify tools are working correctly
"""

import asyncio
import sys
from agent_tools import ToolManager

TEST_CONFIG = {
    "wifi": {"enabled": True},
    "email": {
        "enabled": True,
        "smtp": {
            "server": "smtp.gmail.com",
            "port": 587
        }
    },
    "browser": {"enabled": True},
    "system": {
        "enabled": True,
        "allowed_commands": ["ls", "pwd", "date", "whoami", "uptime"]
    },
    "files": {
        "enabled": True,
        "allowed_directories": ["~/Documents", "~/Downloads", "/tmp"]
    }
}


async def test_wifi():
    """Test WiFi tool"""
    print("\n" + "="*50)
    print("Testing WiFi Tool")
    print("="*50)
    
    tool_manager = ToolManager(TEST_CONFIG)
    
    # Test status
    print("\n1. Checking WiFi status...")
    result = await tool_manager.execute_tool("wifi", action="status")
    print(f"Result: {result}")
    
    # Test list
    print("\n2. Listing WiFi networks...")
    result = await tool_manager.execute_tool("wifi", action="list")
    if result.get("success"):
        print(f"Found {result.get('count', 0)} networks")
        for net in result.get("networks", [])[:3]:
            print(f"  - {net['ssid']} (Signal: {net['signal']}%, Security: {net['security']})")
    else:
        print(f"Error: {result.get('error')}")


async def test_browser():
    """Test Browser tool"""
    print("\n" + "="*50)
    print("Testing Browser Tool")
    print("="*50)
    
    tool_manager = ToolManager(TEST_CONFIG)
    
    print("\nOpening Google (will open in your browser)...")
    print("Press Ctrl+C within 3 seconds to skip...")
    try:
        await asyncio.sleep(3)
        result = await tool_manager.execute_tool("browser", url="google.com")
        print(f"Result: {result}")
    except KeyboardInterrupt:
        print("\nSkipped browser test")


async def test_system():
    """Test System command tool"""
    print("\n" + "="*50)
    print("Testing System Command Tool")
    print("="*50)
    
    tool_manager = ToolManager(TEST_CONFIG)
    
    commands = [
        ("date", []),
        ("whoami", []),
        ("pwd", []),
    ]
    
    for cmd, args in commands:
        print(f"\nExecuting: {cmd}")
        result = await tool_manager.execute_tool("system", command=cmd, args=args)
        if result.get("success"):
            print(f"Output: {result.get('output', '').strip()}")
        else:
            print(f"Error: {result.get('error')}")


async def test_files():
    """Test File tool"""
    print("\n" + "="*50)
    print("Testing File Tool")
    print("="*50)
    
    tool_manager = ToolManager(TEST_CONFIG)
    
    # Test list
    print("\nListing files in /tmp...")
    result = await tool_manager.execute_tool("files", action="list", path="/tmp")
    if result.get("success"):
        files = result.get("files", [])
        print(f"Found {len(files)} items:")
        for f in files[:5]:
            print(f"  - {f['name']} ({f['type']})")
    else:
        print(f"Error: {result.get('error')}")
    
    # Test write/read/delete
    print("\nTesting write/read/delete...")
    test_file = "/tmp/agent_test.txt"
    test_content = "Hello from agent tools!"
    
    # Write
    print(f"Writing to {test_file}...")
    result = await tool_manager.execute_tool("files", action="write", path=test_file, content=test_content)
    print(f"Write result: {result.get('success')}")
    
    # Read
    print(f"Reading {test_file}...")
    result = await tool_manager.execute_tool("files", action="read", path=test_file)
    if result.get("success"):
        print(f"Content: {result.get('content')}")
    
    # Delete
    print(f"Deleting {test_file}...")
    result = await tool_manager.execute_tool("files", action="delete", path=test_file)
    print(f"Delete result: {result.get('success')}")


async def test_email():
    """Test Email tool"""
    print("\n" + "="*50)
    print("Testing Email Tool")
    print("="*50)
    
    print("\nEmail tool requires EMAIL_USER and EMAIL_PASSWORD environment variables")
    print("Skipping actual send test (configure credentials to test)")
    
    tool_manager = ToolManager(TEST_CONFIG)
    print(f"Email tool registered: {'email' in tool_manager.list_tools()}")


async def test_tool_detection():
    """Test natural language tool detection"""
    print("\n" + "="*50)
    print("Testing Natural Language Detection")
    print("="*50)
    
    tool_manager = ToolManager(TEST_CONFIG)
    
    test_phrases = [
        "Show me WiFi networks",
        "What's my WiFi status?",
        "Open Google",
        "List files in my Documents",
        "Send an email",
        "What time is it?",
    ]
    
    for phrase in test_phrases:
        result = tool_manager.parse_tool_request(phrase)
        print(f"\n'{phrase}'")
        print(f"  → Detected: {result}")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  Agent Tools Test Suite")
    print("="*60)
    
    try:
        # Test tool detection
        await test_tool_detection()
        
        # Test WiFi
        await test_wifi()
        
        # Test system commands
        await test_system()
        
        # Test files
        await test_files()
        
        # Test email info
        await test_email()
        
        # Test browser (with option to skip)
        await test_browser()
        
        print("\n" + "="*60)
        print("✓ All tests completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

