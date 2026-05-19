#!/usr/bin/env python3
"""
Agent Tools - System capabilities for the voice agent
Provides tools for WiFi, email, web browsing, file operations, etc.
"""

import os
import subprocess
import logging
import json
import smtplib
import webbrowser
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("vchat-agent.tools")


# ============================================================
# Base Tool Class
# ============================================================

class Tool(ABC):
    """Base class for agent tools"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.enabled = True
    
    @abstractmethod
    def get_description(self) -> str:
        """Return tool description for LLM context"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    def to_schema(self) -> Dict[str, Any]:
        """Return JSON schema for this tool"""
        return {
            "name": self.name,
            "description": self.get_description(),
            "enabled": self.enabled
        }


# ============================================================
# WiFi Management Tool
# ============================================================

class WiFiTool(Tool):
    """Manage WiFi connections"""
    
    def get_description(self) -> str:
        return """WiFi management tool. Can:
- List available networks: wifi_list_networks
- Connect to network: wifi_connect(ssid, password)
- Disconnect: wifi_disconnect
- Get current status: wifi_status"""
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute WiFi actions
        
        Args:
            action: One of 'list', 'connect', 'disconnect', 'status'
            ssid: Network name (for connect)
            password: Network password (for connect)
        """
        try:
            if action == "list":
                return await self._list_networks()
            elif action == "connect":
                ssid = kwargs.get("ssid")
                password = kwargs.get("password")
                if not ssid:
                    return {"success": False, "error": "SSID required"}
                return await self._connect(ssid, password)
            elif action == "disconnect":
                return await self._disconnect()
            elif action == "status":
                return await self._get_status()
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            logger.error(f"WiFi tool error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _list_networks(self) -> Dict[str, Any]:
        """List available WiFi networks"""
        try:
            # Use nmcli on Linux
            result = subprocess.run(
                ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                networks = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            networks.append({
                                "ssid": parts[0],
                                "signal": parts[1] if len(parts) > 1 else "0",
                                "security": parts[2] if len(parts) > 2 else "Open"
                            })
                
                return {
                    "success": True,
                    "networks": networks[:10],  # Top 10
                    "count": len(networks)
                }
            else:
                return {"success": False, "error": result.stderr}
                
        except FileNotFoundError:
            return {"success": False, "error": "nmcli not found. WiFi management requires NetworkManager."}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _connect(self, ssid: str, password: Optional[str] = None) -> Dict[str, Any]:
        """Connect to WiFi network"""
        try:
            cmd = ["nmcli", "device", "wifi", "connect", ssid]
            if password:
                cmd.extend(["password", password])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Connected to {ssid}",
                    "ssid": ssid
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or "Connection failed"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _disconnect(self) -> Dict[str, Any]:
        """Disconnect from WiFi"""
        try:
            result = subprocess.run(
                ["nmcli", "device", "disconnect", "wlan0"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {"success": True, "message": "WiFi disconnected"}
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_status(self) -> Dict[str, Any]:
        """Get WiFi connection status"""
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "device", "status"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(':')
                    if len(parts) >= 3 and 'wlan' in parts[0]:
                        return {
                            "success": True,
                            "device": parts[0],
                            "state": parts[1],
                            "connection": parts[2] if parts[2] else "Not connected"
                        }
                
                return {"success": True, "state": "unavailable"}
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================
# Email Tool
# ============================================================

class EmailTool(Tool):
    """Send emails via SMTP"""
    
    def __init__(self, smtp_config: Optional[Dict[str, str]] = None):
        super().__init__()
        self.smtp_config = smtp_config or {}
        self.smtp_server = self.smtp_config.get("server", "smtp.gmail.com")
        self.smtp_port = int(self.smtp_config.get("port", "587"))
        self.username = self.smtp_config.get("username", os.getenv("EMAIL_USER", ""))
        self.password = self.smtp_config.get("password", os.getenv("EMAIL_PASSWORD", ""))
        self.from_email = self.smtp_config.get("from_email", self.username)
    
    def get_description(self) -> str:
        return """Email tool. Can send emails via SMTP.
Usage: send_email(to, subject, body)
Requires EMAIL_USER and EMAIL_PASSWORD environment variables or config."""
    
    async def execute(self, to: str, subject: str, body: str, **kwargs) -> Dict[str, Any]:
        """Send an email
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
        """
        if not self.username or not self.password:
            return {
                "success": False,
                "error": "Email credentials not configured. Set EMAIL_USER and EMAIL_PASSWORD."
            }
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"✓ Email sent to {to}")
            return {
                "success": True,
                "message": f"Email sent to {to}",
                "to": to,
                "subject": subject
            }
            
        except Exception as e:
            logger.error(f"Email error: {e}")
            return {"success": False, "error": str(e)}


# ============================================================
# Web Browser Tool
# ============================================================

class BrowserTool(Tool):
    """Open websites in browser"""
    
    def get_description(self) -> str:
        return """Web browser tool. Can open websites.
Usage: open_browser(url)
Example: open_browser('https://google.com')"""
    
    async def execute(self, url: str, **kwargs) -> Dict[str, Any]:
        """Open URL in default browser
        
        Args:
            url: URL to open
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            webbrowser.open(url)
            logger.info(f"✓ Opened {url} in browser")
            
            return {
                "success": True,
                "message": f"Opened {url}",
                "url": url
            }
            
        except Exception as e:
            logger.error(f"Browser error: {e}")
            return {"success": False, "error": str(e)}


# ============================================================
# System Command Tool
# ============================================================

class SystemCommandTool(Tool):
    """Execute safe system commands"""
    
    def __init__(self, allowed_commands: Optional[List[str]] = None):
        super().__init__()
        # Whitelist of allowed commands for security
        self.allowed_commands = allowed_commands or [
            "ls", "pwd", "date", "whoami", "uptime",
            "df", "free", "ps", "top", "htop",
            "ip", "ping", "hostname"
        ]
    
    def get_description(self) -> str:
        return f"""System command tool. Can execute safe system commands.
Allowed commands: {', '.join(self.allowed_commands)}
Usage: run_command(command, args)"""
    
    async def execute(self, command: str, args: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute a whitelisted system command
        
        Args:
            command: Command to execute
            args: Command arguments
        """
        if command not in self.allowed_commands:
            return {
                "success": False,
                "error": f"Command '{command}' not allowed. Allowed: {', '.join(self.allowed_commands)}"
            }
        
        try:
            cmd = [command]
            if args:
                cmd.extend(args)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "exit_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Command error: {e}")
            return {"success": False, "error": str(e)}


# ============================================================
# File Operations Tool
# ============================================================

class FileTool(Tool):
    """Safe file operations"""
    
    def __init__(self, allowed_directories: Optional[List[str]] = None):
        super().__init__()
        # Restrict to safe directories
        self.allowed_directories = allowed_directories or [
            str(Path.home() / "Documents"),
            str(Path.home() / "Downloads"),
            "/tmp"
        ]
    
    def get_description(self) -> str:
        return """File operations tool. Can:
- List files: list_files(directory)
- Read file: read_file(path)
- Write file: write_file(path, content)
- Delete file: delete_file(path)
Restricted to safe directories."""
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if path is in allowed directories"""
        path = Path(path).resolve()
        return any(str(path).startswith(allowed) for allowed in self.allowed_directories)
    
    async def execute(self, action: str, path: str = None, content: str = None, **kwargs) -> Dict[str, Any]:
        """Execute file operation
        
        Args:
            action: One of 'list', 'read', 'write', 'delete'
            path: File or directory path
            content: Content for write operation
        """
        try:
            if action == "list":
                if not path:
                    path = self.allowed_directories[0]
                return await self._list_files(path)
            elif action == "read":
                if not path:
                    return {"success": False, "error": "Path required"}
                return await self._read_file(path)
            elif action == "write":
                if not path or content is None:
                    return {"success": False, "error": "Path and content required"}
                return await self._write_file(path, content)
            elif action == "delete":
                if not path:
                    return {"success": False, "error": "Path required"}
                return await self._delete_file(path)
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            logger.error(f"File tool error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _list_files(self, directory: str) -> Dict[str, Any]:
        """List files in directory"""
        if not self._is_safe_path(directory):
            return {"success": False, "error": "Directory not allowed"}
        
        try:
            path = Path(directory)
            if not path.exists():
                return {"success": False, "error": "Directory not found"}
            
            files = []
            for item in path.iterdir():
                files.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            
            return {
                "success": True,
                "directory": str(path),
                "files": files[:50],  # Limit to 50
                "count": len(files)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _read_file(self, filepath: str) -> Dict[str, Any]:
        """Read file content"""
        if not self._is_safe_path(filepath):
            return {"success": False, "error": "File not allowed"}
        
        try:
            path = Path(filepath)
            if not path.exists():
                return {"success": False, "error": "File not found"}
            
            # Limit file size
            if path.stat().st_size > 1_000_000:  # 1MB
                return {"success": False, "error": "File too large (max 1MB)"}
            
            content = path.read_text()
            return {
                "success": True,
                "path": str(path),
                "content": content,
                "size": len(content)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _write_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        if not self._is_safe_path(filepath):
            return {"success": False, "error": "File path not allowed"}
        
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            
            return {
                "success": True,
                "path": str(path),
                "size": len(content)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _delete_file(self, filepath: str) -> Dict[str, Any]:
        """Delete file"""
        if not self._is_safe_path(filepath):
            return {"success": False, "error": "File path not allowed"}
        
        try:
            path = Path(filepath)
            if not path.exists():
                return {"success": False, "error": "File not found"}
            
            path.unlink()
            return {
                "success": True,
                "path": str(path),
                "message": "File deleted"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================
# Tool Manager
# ============================================================

class ToolManager:
    """Manages all available tools for the agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all available tools"""
        # WiFi
        if self.config.get("wifi", {}).get("enabled", True):
            self.tools["wifi"] = WiFiTool()
        
        # Email
        if self.config.get("email", {}).get("enabled", True):
            smtp_config = self.config.get("email", {}).get("smtp", {})
            self.tools["email"] = EmailTool(smtp_config)
        
        # Browser
        if self.config.get("browser", {}).get("enabled", True):
            self.tools["browser"] = BrowserTool()
        
        # System commands
        if self.config.get("system", {}).get("enabled", True):
            allowed_cmds = self.config.get("system", {}).get("allowed_commands")
            self.tools["system"] = SystemCommandTool(allowed_cmds)
        
        # File operations
        if self.config.get("files", {}).get("enabled", True):
            allowed_dirs = self.config.get("files", {}).get("allowed_directories")
            self.tools["files"] = FileTool(allowed_dirs)
        
        logger.info(f"✓ Registered {len(self.tools)} tools: {list(self.tools.keys())}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools for LLM"""
        if not self.tools:
            return "No tools available."
        
        descriptions = ["Available tools:"]
        for name, tool in self.tools.items():
            descriptions.append(f"\n{name.upper()}:")
            descriptions.append(f"  {tool.get_description()}")
        
        return "\n".join(descriptions)
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found. Available: {', '.join(self.list_tools())}"
            }
        
        if not tool.enabled:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is disabled"
            }
        
        logger.info(f"🔧 Executing tool: {tool_name} with {kwargs}")
        result = await tool.execute(**kwargs)
        logger.info(f"✓ Tool result: {result}")
        
        return result
    
    def parse_tool_request(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tool request from natural language (simple keyword matching)"""
        text_lower = text.lower()
        
        # WiFi patterns
        if any(kw in text_lower for kw in ["wifi", "wi-fi", "network", "internet"]):
            if any(kw in text_lower for kw in ["connect", "join"]):
                return {"tool": "wifi", "action": "connect"}
            elif any(kw in text_lower for kw in ["disconnect", "turn off"]):
                return {"tool": "wifi", "action": "disconnect"}
            elif any(kw in text_lower for kw in ["list", "show", "available", "scan"]):
                return {"tool": "wifi", "action": "list"}
            elif any(kw in text_lower for kw in ["status", "connected", "connection"]):
                return {"tool": "wifi", "action": "status"}
        
        # Email patterns
        if any(kw in text_lower for kw in ["email", "send email", "mail", "send mail"]):
            return {"tool": "email", "action": "send"}
        
        # Browser patterns
        if any(kw in text_lower for kw in ["open", "browse", "website", "google", "search"]):
            if any(kw in text_lower for kw in ["google", "search"]):
                return {"tool": "browser", "action": "search"}
            return {"tool": "browser", "action": "open"}
        
        # File patterns
        if any(kw in text_lower for kw in ["file", "folder", "directory"]):
            if any(kw in text_lower for kw in ["list", "show"]):
                return {"tool": "files", "action": "list"}
            elif any(kw in text_lower for kw in ["read", "open"]):
                return {"tool": "files", "action": "read"}
            elif any(kw in text_lower for kw in ["write", "create", "save"]):
                return {"tool": "files", "action": "write"}
            elif any(kw in text_lower for kw in ["delete", "remove"]):
                return {"tool": "files", "action": "delete"}
        
        return None

