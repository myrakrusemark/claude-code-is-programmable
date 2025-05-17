import json
import subprocess
from typing import List, Optional


def run_claude(
    prompt: str,
    output_format: str = "json",
    allowed_tools: Optional[List[str]] = None,
    cli: str = "claude",
) -> str:
    """Run Claude Code in headless mode with the given output format."""
    cmd = [cli, "-p", prompt, "--output-format", output_format]
    if allowed_tools:
        cmd.extend(["--allowedTools", *allowed_tools])
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude failed: {result.stderr}")
    return result.stdout


def run_claude_json(
    prompt: str,
    allowed_tools: Optional[List[str]] = None,
    cli: str = "claude",
) -> dict:
    """Run Claude and parse JSON output."""
    output = run_claude(prompt, "json", allowed_tools, cli)
    return json.loads(output)
