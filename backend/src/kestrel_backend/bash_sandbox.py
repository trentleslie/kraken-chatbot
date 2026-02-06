"""Bash command sandboxing for security.

This module provides security controls for the Bash tool to prevent prompt
injection attacks and malicious command execution. It implements a PreToolUse
hook that validates all Bash commands before execution.

Security layers:
1. Command Whitelist - Only known-safe executables allowed
2. Network Isolation - ALL network tools blocked (use MCP tools instead)
3. Dangerous Patterns - Destructive command patterns blocked
4. Path Restrictions - Sensitive paths inaccessible
5. Pipe Injection - Shell injection via pipes blocked
6. Execution Timeout - 30-second max execution time
"""

import re
import shlex
from typing import Any

from claude_agent_sdk.types import (
    HookContext,
    HookJSONOutput,
    PreToolUseHookInput,
    PreToolUseHookSpecificOutput,
)

# Execution timeout in seconds
BASH_TIMEOUT_SECONDS = 30

# Commands allowed for data processing (NO NETWORK TOOLS)
ALLOWED_COMMANDS = frozenset([
    # Data processing
    "jq", "python", "python3", "node",
    # Text processing
    "grep", "awk", "sed", "sort", "uniq", "wc", "head", "tail", "cat",
    # File info (read-only)
    "ls", "find", "file", "stat", "du",
    # Other safe utilities
    "echo", "printf", "date", "true", "false",
])

# Network tools - EXPLICITLY BLOCKED (agent has MCP tools for network)
BLOCKED_NETWORK_COMMANDS = frozenset([
    "curl", "wget", "nc", "netcat", "ncat",
    "ssh", "scp", "sftp", "rsync",
    "ftp", "tftp",
    "telnet", "nmap", "ping",
    "dig", "nslookup", "host",
    "socat", "openssl",  # can be used for network connections
])

# Patterns that indicate dangerous commands
DANGEROUS_PATTERNS = [
    r"\brm\s+(-[rf]+\s+)*[/~]",       # rm with paths
    r"\bdd\s+",                        # dd command
    r"\bmkfs\b",                       # filesystem creation
    r"\bformat\b",                     # format command
    r"\bshutdown\b|\breboot\b",        # system control
    r"\bsudo\b|\bdoas\b",              # privilege escalation
    r"\bchmod\s+[0-7]*777\b",          # world-writable
    r"\bchown\b",                      # ownership changes
    r">\s*/dev/",                      # writing to devices
    r"\|.*\b(ba)?sh\b",                # any pipe to shell
    r"\beval\b",                       # eval command
    r"\bexec\b",                       # exec command
    r"`.*`",                           # backtick command substitution
    r"\$\(.*\)",                       # $() command substitution (in args)
    r";\s*rm\b",                       # command chaining with rm
    r"&&\s*rm\b",                      # conditional rm
    r"\|\|\s*rm\b",                    # fallback rm
    r"/dev/tcp/",                      # bash network pseudofiles
    r"/dev/udp/",                      # bash network pseudofiles
]

# Paths that should never be accessed
BLOCKED_PATHS = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "~/.ssh", "~/.aws", "~/.config",
    "/root", "/home/ubuntu/.ssh",
    ".env", ".git/config",
]


def validate_bash_command(command: str) -> tuple[bool, str]:
    """
    Validate a bash command for safety.

    This function checks the command against multiple security layers:
    1. Empty command check
    2. Dangerous pattern detection (e.g., rm -rf, pipe to shell)
    3. Blocked path access prevention
    4. Network command blocking
    5. Command whitelist enforcement

    Args:
        command: The bash command string to validate

    Returns:
        A tuple of (is_allowed, reason):
        - is_allowed: True if the command is safe to execute
        - reason: Explanation of why the command was allowed or blocked
    """
    if not command or not command.strip():
        return False, "Empty command"

    command = command.strip()

    # Check for dangerous patterns first
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Dangerous pattern detected: {pattern}"

    # Check for blocked paths
    for blocked_path in BLOCKED_PATHS:
        if blocked_path in command:
            return False, f"Access to {blocked_path} is not allowed"

    # Extract the base command
    try:
        parts = shlex.split(command)
        if not parts:
            return False, "Could not parse command"
        base_cmd = parts[0].split("/")[-1]  # Handle full paths
    except ValueError as e:
        return False, f"Invalid command syntax: {e}"

    # Check for network commands (NEVER allowed - use MCP tools instead)
    if base_cmd in BLOCKED_NETWORK_COMMANDS:
        return False, f"Network command '{base_cmd}' is blocked. Use MCP tools for network operations."

    # Also check if any network command appears anywhere in the command (piped, etc.)
    for net_cmd in BLOCKED_NETWORK_COMMANDS:
        if re.search(rf"\b{net_cmd}\b", command):
            return False, f"Network command '{net_cmd}' detected. Network access via Bash is disabled."

    # Check if base command is in whitelist
    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"Command '{base_cmd}' is not in the allowed list. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"

    return True, "Command approved"


def wrap_with_timeout(command: str) -> str:
    """
    Wrap a command with timeout to prevent hanging.

    Uses the 'timeout' command to enforce an execution time limit,
    protecting against denial-of-service attacks via infinite loops
    or hanging processes.

    Args:
        command: The command to wrap with a timeout

    Returns:
        The command wrapped with timeout enforcement
    """
    # Use 'timeout' command to enforce execution limit
    # --signal=KILL ensures the process is killed if it doesn't respond to TERM
    return f"timeout --signal=KILL {BASH_TIMEOUT_SECONDS}s {command}"


async def bash_security_hook(
    input_data: PreToolUseHookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    PreToolUse hook for Bash command validation.

    This hook intercepts Bash tool calls, validates the command against
    security rules, and wraps approved commands with a timeout.

    Args:
        input_data: The hook input containing tool_name, tool_input, etc.
        tool_use_id: The unique identifier for this tool use
        context: Hook context with abort signal support

    Returns:
        HookJSONOutput with either:
        - permissionDecision="allow" and wrapped command for safe commands
        - permissionDecision="deny" and reason for blocked commands
    """
    # Only process Bash tool calls
    if input_data.get("tool_name") != "Bash":
        return {}

    command = input_data.get("tool_input", {}).get("command", "")

    is_allowed, reason = validate_bash_command(command)

    if is_allowed:
        # Wrap command with timeout for safety
        wrapped_command = wrap_with_timeout(command)

        hook_output: PreToolUseHookSpecificOutput = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": f"{reason} (timeout: {BASH_TIMEOUT_SECONDS}s)",
            "updatedInput": {
                **input_data.get("tool_input", {}),
                "command": wrapped_command
            }
        }

        return {
            "hookSpecificOutput": hook_output
        }
    else:
        hook_output = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason
        }

        return {
            "systemMessage": f"Bash command blocked: {reason}",
            "hookSpecificOutput": hook_output
        }
