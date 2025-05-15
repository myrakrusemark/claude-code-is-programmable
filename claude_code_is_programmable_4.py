#!/usr/bin/env -S uv run --script
"""
# Programmable Claude Code script that allows different output formats

Usage:
    uv run claude_code_is_programmable_4.py
    uv run claude_code_is_programmable_4.py --output-format text
    uv run claude_code_is_programmable_4.py --output-format json
    uv run claude_code_is_programmable_4.py --output-format stream-json

The script passes the selected output format directly to the Claude Code CLI using
the --output-format flag, which controls how Claude's responses are formatted.
"""

import subprocess
import json
import argparse


def output_text(content):
    """Output in plain text format"""
    print(content)


def output_json(content):
    """Output in JSON format"""
    try:
        # If content is already JSON-formatted string, parse it first
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            # If not valid JSON, wrap as a simple message object
            print(json.dumps({"message": content}, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))


def output_stream_json(content):
    """Output in streaming JSON format (one JSON object per line)"""
    try:
        # If content is already JSON-formatted string, parse it first
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed))
        except json.JSONDecodeError:
            # If not valid JSON, wrap as a simple message object
            print(json.dumps({"message": content}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


def main():
    parser = argparse.ArgumentParser(description="Claude Code runner script")
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Specify output format",
    )
    args = parser.parse_args()

    prompt = """
    RUN the following commands:
    1. Run git status
    2. Add a brief comment about the changes at the top of claude_code_is_programmable_4.py
    3. Summarize what you've done
    """

    # Build command with appropriate output format flag
    command = ["claude", "-p", prompt, "--allowedTools", "Edit", "Bash", "Write"]

    # Add the output format flag if specified
    if args.output_format == "json":
        command.extend(["--output-format", "json"])
    elif args.output_format == "stream-json":
        command.extend(["--output-format", "stream-json"])

    process = subprocess.run(command, capture_output=True, text=True, check=True)

    output = process.stdout or "No output captured"

    # Output the result directly since Claude CLI already formats it
    # We'll only use our formatting functions for text mode or if needed for processing
    if args.output_format == "text":
        output_text(output)
    else:
        # For JSON modes, Claude CLI already returns formatted output
        print(output)


if __name__ == "__main__":
    main()
