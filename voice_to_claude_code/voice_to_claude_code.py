#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "RealtimeSTT",
#   "openai",
#   "python-dotenv",
#   "rich",
#   "numpy",
#   "sounddevice",
#   "soundfile",
# ]
# ///

"""
# Voice to Claude Code

A voice-enabled Claude Code assistant that allows you to interact with Claude Code using voice commands.
This tool combines RealtimeSTT for speech recognition and OpenAI TTS for speech output.

## Features
- Real-time speech recognition using RealtimeSTT
- Claude Code integration for programmable AI coding
- Text-to-speech responses using OpenAI TTS
- Conversation history tracking
- Voice trigger activation

## Requirements
- OpenAI API key (for TTS)
- Anthropic API key (for Claude Code)
- Python 3.9+
- UV package manager (for dependency management)

## Usage
Run the script:
```bash
./voice_to_claude_code.py
```

Speak to the assistant using the trigger word "claude" in your query.
For example: "Hey claude, create a simple hello world script"

Press Ctrl+C to exit.
"""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import List, Dict, Any, Optional, Union
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
from rich.syntax import Syntax
from dotenv import load_dotenv
import openai
from openai import OpenAI
from RealtimeSTT import AudioToTextRecorder
import logging

# Configuration - default values
TRIGGER_WORD = "claude"
STT_MODEL = "small.en"  # Options: tiny.en, base.en, small.en, medium.en, large-v2
TTS_VOICE = "nova"  # Options: alloy, echo, fable, onyx, nova, shimmer
DEFAULT_CLAUDE_TOOLS = ["Bash", "Edit", "Write", "GlobTool", "GrepTool", "LSTool"]

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("claude_code_assistant")

# Suppress RealtimeSTT logs and all related loggers
logging.getLogger("RealtimeSTT").setLevel(logging.ERROR)
logging.getLogger("transcribe").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)
logging.getLogger("audio_recorder").setLevel(logging.ERROR)
logging.getLogger("whisper").setLevel(logging.ERROR)
logging.getLogger("faster_whisper.transcribe").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

console = Console()

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    console.print(
        f"[bold red]Error: Missing required environment variables: {', '.join(missing_vars)}[/bold red]"
    )
    console.print("Please set these in your .env file or as environment variables.")
    sys.exit(1)

# Initialize OpenAI client for TTS
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def invoke_claude_code(prompt: str) -> dict:
    """
    Invokes Claude Code programmatically with a specific prompt.
    All tools are available by default and output is streamed in real-time.

    Args:
        prompt (str): The instruction to send to Claude Code

    Returns:
        dict: Response containing success status, message, and error
    """
    log.info("=" * 60)
    log.info(f"invoke_claude_code(prompt_length={len(prompt)})")

    full_output = []  # To collect all output

    try:
        # Use the constant but don't pass it to Claude (allowing all tools by default)
        log.info(f"Default tools list (for reference): {DEFAULT_CLAUDE_TOOLS}")
        log.info("Using all available tools (not restricted)")

        # Set up command with streaming output format
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json"
        ]
        log.info(f"Command: {' '.join(cmd[:3])}... (truncated)")

        # Execute Claude Code as a subprocess with line buffering
        log.info("Starting Claude Code subprocess with streaming output...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Process and display streaming output in real-time
        console.print("\n[bold blue]🔄 Streaming Claude Code output:[/bold blue]")

        from rich.syntax import Syntax

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            # Display the streaming output
            try:
                syntax = Syntax(line, "json", theme="monokai", line_numbers=False)
                console.print(syntax)
            except Exception:
                # Fall back to plain output if syntax highlighting fails
                console.print(line.strip())

            # Collect the output
            full_output.append(line)

        # Read any errors
        stderr = process.stderr.read()

        # Get return code and check result
        return_code = process.wait()
        log.info(f"Process finished with return code: {return_code}")

        if return_code == 0:
            log.info(f"Claude Code succeeded, collected {len(full_output)} output lines")
            console.print("[bold green]✅ Claude Code completed successfully[/bold green]")
            return {
                "success": True,
                "message": "".join(full_output),
                "error": None
            }
        else:
            error_msg = f"Claude Code failed with exit code: {return_code}"
            log.error(f"{error_msg}\nError: {stderr[:500]}...")
            console.print(f"[bold red]❌ {error_msg}[/bold red]")
            return {
                "success": False,
                "message": "Claude Code execution failed",
                "error": stderr,
            }

    except Exception as e:
        error_msg = f"Error in invoke_claude_code: {str(e)}"
        log.error(error_msg)
        console.print(f"[bold red]❌ {error_msg}[/bold red]")
        return {
            "success": False,
            "message": "Claude Code execution failed",
            "error": error_msg,
        }
    finally:
        log.info("=" * 60)


class ClaudeCodeAssistant:
    def __init__(self, initial_prompt: Optional[str] = None):
        log.info("Initializing Claude Code Assistant")
        self.recorder = None
        self.initial_prompt = initial_prompt
        self.first_prompt = True  # Initialize first_prompt flag as true for first run

        # Set up recorder only
        self.setup_recorder()

    def setup_recorder(self):
        """Set up the RealtimeSTT recorder"""
        log.info(f"Setting up STT recorder with model {STT_MODEL}")

        self.recorder = AudioToTextRecorder(
            model=STT_MODEL,
            language="en",
            compute_type="float32",
            post_speech_silence_duration=0.8,
            beam_size=5,
            initial_prompt=None,
            spinner=False,
            print_transcription_time=False,
            enable_realtime_transcription=True,
            realtime_model_type="tiny.en",
            realtime_processing_pause=0.4,
        )

        log.info(f"STT recorder initialized with model {STT_MODEL}")

    # Removed format_conversation_history method as we no longer track history

    async def listen(self) -> str:
        """Listen for user speech and convert to text"""
        log.info("Listening for speech...")

        # If this is the first call and we have an initial prompt, use it instead of recording
        if hasattr(self, "initial_prompt") and self.initial_prompt:
            prompt = self.initial_prompt

            # Display the prompt as if it were spoken
            console.print(
                Panel(title="You", title_align="left", renderable=Markdown(prompt))
            )

            # Clear the initial prompt so it's only used once
            self.initial_prompt = None

            return prompt

        # Set up realtime display
        def on_realtime_update(text):
            # Clear line and update realtime text
            sys.stdout.write("\r\033[K")  # Clear line
            sys.stdout.write(f"Listening: {text}")
            sys.stdout.flush()

        self.recorder.on_realtime_transcription_update = on_realtime_update

        # Create a synchronization object for the callback
        result_container = {"text": "", "done": False}

        def callback(text):
            if text:
                console.print("")
                console.print(
                    Panel(title="You", title_align="left", renderable=Markdown(text))
                )
                log.info(f'Heard: "{text}"')
                result_container["text"] = text

            result_container["done"] = True

        # Get text with callback
        self.recorder.text(callback)

        # Wait for result with a simple polling loop
        timeout = 60  # seconds
        wait_interval = 0.1  # seconds
        elapsed = 0

        while not result_container["done"] and elapsed < timeout:
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval

        if elapsed >= timeout:
            log.warning("Timeout waiting for speech")
            return ""

        return result_container["text"]

    async def speak(self, text: str):
        """Convert text to speech using OpenAI TTS"""
        log.info(f'Speaking: "{text[:50]}..."')

        try:
            # Generate speech
            response = client.audio.speech.create(
                model="tts-1",
                voice=TTS_VOICE,
                input=text,
                speed=1.0,
            )

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                response.stream_to_file(temp_filename)

            # Play audio
            data, samplerate = sf.read(temp_filename)
            sd.play(data, samplerate)

            # Log start time for duration tracking
            start_time = asyncio.get_event_loop().time()

            # Wait for audio to finish
            sd.wait()

            # Calculate speech duration
            duration = asyncio.get_event_loop().time() - start_time

            # Clean up the temporary file
            os.unlink(temp_filename)

            log.info(f"Audio played (duration: {duration:.2f}s)")

        except Exception as e:
            log.error(f"Error in speech synthesis: {str(e)}")
            console.print(f"[bold red]Error in speech synthesis:[/bold red] {str(e)}")
            # Display the text as fallback
            console.print(f"[italic yellow]Text:[/italic yellow] {text}")

    async def process_message(self, message: str) -> Optional[str]:
        """Process the user message and run Claude Code"""
        log.info(f'Processing message: "{message}"')

        # Check for trigger word
        if TRIGGER_WORD.lower() not in message.lower():
            log.info("Trigger word not detected, skipping")
            return None

        # For the first prompt, don't use --continue
        # For subsequent prompts, use --continue flag
        if self.first_prompt:
            # First prompt - start a new conversation
            log.info("First prompt - starting new Claude Code session")
            cmd = ["claude", "--output-format", "stream-json", "-p", message]
            self.first_prompt = False  # Set to false after first run
        else:
            # Subsequent prompts - continue existing conversation
            log.info("Using --continue flag to continue previous conversation")
            cmd = ["claude", "--continue", "--output-format", "stream-json", "--print", message]

        # Execute Claude Code as a subprocess with streaming output
        log.info(f"Command: {' '.join(cmd[:3])}... (truncated)")
        log.info("Starting Claude Code subprocess with streaming output...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Collect the final text response
        final_text_response = []
        current_message_content = []

        # Process and display streaming output in real-time
        console.print("\n[bold blue]🔄 Running Claude Code...[/bold blue]")

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            # Skip empty lines
            if not line.strip():
                continue

            try:
                # Parse the JSON line
                json_obj = json.loads(line)

                # Display formatted JSON for debugging
                syntax = Syntax(json.dumps(json_obj, indent=2), "json", theme="monokai", line_numbers=False)
                console.print(syntax)

                # Extract and handle different types of JSON responses
                if json_obj.get("type") == "message" and json_obj.get("role") == "assistant":
                    # Process assistant message
                    if "content" in json_obj:
                        for content_item in json_obj["content"]:
                            if content_item.get("type") == "text":
                                # Extract text content
                                current_message_content.append(content_item.get("text", ""))
                                # Display the text content to the user
                                console.print(f"[bold green]Claude:[/bold green] {content_item.get('text', '')}")

                # When we get a system cost message, we know the interaction is complete
                if json_obj.get("role") == "system" and "result" in json_obj:
                    final_result = json_obj.get("result", "")
                    if final_result:
                        final_text_response.append(final_result)

            except json.JSONDecodeError as e:
                # Not valid JSON, just display the raw line
                log.warning(f"Invalid JSON: {line.strip()}")
                console.print(f"[yellow]{line.strip()}[/yellow]")

        # Read any errors
        stderr = process.stderr.read()

        # Get return code and check result
        return_code = process.wait()
        log.info(f"Process finished with return code: {return_code}")

        if return_code == 0:
            # Combine all text content into a single response
            # Use current_message_content if available, otherwise use final_text_response
            if current_message_content:
                response = "\n".join(current_message_content)
            else:
                response = "\n".join(final_text_response)

            log.info(f"Claude Code succeeded, extracted text response of length: {len(response)}")

            # Display the final response
            if response:
                console.print(Panel(title="Claude Final Response", renderable=Markdown(response)))

            return response
        else:
            error_msg = f"Claude Code failed with exit code: {return_code}"
            log.error(f"{error_msg}\nError: {stderr[:500]}...")

            error_response = "I'm sorry, but I encountered an error while processing your request. Please try again."
            return error_response

    async def conversation_loop(self):
        """Run the main conversation loop"""
        log.info("Starting conversation loop")

        console.print(
            Panel.fit(
                "[bold magenta]🎤 Claude Code Voice Assistant Ready[/bold magenta]\n"
                f"Speak to interact. Include the word '{TRIGGER_WORD}' to activate.\n"
                f"The assistant will listen, process with Claude Code, and respond using voice '{TTS_VOICE}'.\n"
                f"STT model: {STT_MODEL}\n"
                f"Press Ctrl+C to exit."
            )
        )

        try:
            while True:
                user_text = await self.listen()

                if not user_text:
                    console.print("[yellow]No speech detected. Try again.[/yellow]")
                    continue

                response = await self.process_message(user_text)

                # Only speak if we got a response (trigger word was detected)
                if response:
                    await self.speak(response)
                    # Give a small break between interactions
                    await asyncio.sleep(0.5)
                else:
                    # If no trigger word, just continue listening
                    console.print(
                        "[yellow]No trigger word detected. Continuing to listen...[/yellow]"
                    )

        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping assistant...[/bold red]")
            log.info("Conversation loop stopped by keyboard interrupt")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            log.error(f"Error in conversation loop: {str(e)}", exc_info=True)
        finally:
            # Safe cleanup
            try:
                if hasattr(self, "recorder") and self.recorder:
                    # Shutdown the recorder properly
                    self.recorder.shutdown()
            except Exception as shutdown_error:
                log.error(f"Error during shutdown: {str(shutdown_error)}")

            console.print("[bold red]Assistant stopped.[/bold red]")
            log.info("Conversation loop ended")


async def main():
    """Main entry point for the assistant"""
    log.info("Starting Claude Code Voice Assistant")

    # Create assistant instance
    assistant = ClaudeCodeAssistant()

    # Run the conversation loop
    await assistant.conversation_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Program terminated by user")
        console.print("\n[bold red]Program terminated by user.[/bold red]")
