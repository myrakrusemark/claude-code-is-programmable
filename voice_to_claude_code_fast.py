#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "faster-whisper",
#   "python-dotenv",
#   "rich",
#   "numpy",
#   "sounddevice",
#   "soundfile",
#   "markdown",
#   "pvporcupine>=3.0.0",
#   "tqdm>=4.66.0,<4.67.0",
#   "anthropic",
#   "webrtcvad",
# ]
# ///

# Fast voice-enabled assistant using Piper TTS for quick local speech synthesis

"""
# Voice to Claude Code (Fast)

A fast voice-enabled Claude Code assistant using Piper TTS for local speech synthesis,
faster-whisper for speech recognition, and Porcupine wake word detection. Efficient and runs locally!

## Features
- Custom "hey daisy" wake word detection using Porcupine (prevents false activations)
- Real-time speech recognition using faster-whisper
- Voice Activity Detection (VAD) for natural speech boundaries
- Claude Code integration for programmable AI coding
- Fast local text-to-speech using Piper
- Conversation history tracking
- No OpenAI API needed (only Anthropic for Claude)

## Requirements
- Piper TTS installed (with en_US-amy-low.onnx model)
- Anthropic API key (for Claude Code)
- Python 3.9+
- UV package manager (for dependency management)
- Custom "hey daisy" wake word file (hey-daisy_en_linux_v3_0_0.ppn)

## Setup

### 1. Piper TTS
Install Piper and download the model:
```bash
# Install piper
pip install piper-tts

# Or download binary from: https://github.com/rhasspy/piper/releases
```

The script expects Piper to be available as `piper` in PATH.

### 2. Wake Word File
Place the hey-daisy_en_linux_v3_0_0.ppn file in /home/myra/claude-assistant/
The script will automatically load it on startup.

## Usage
Run the script:
```bash
./voice_to_claude_code_fast.py
```

**How to use:**
1. Say "**hey daisy**" (wake word) to activate
2. Wait for confirmation message
3. Speak your request naturally
4. The assistant will process and respond

Optional arguments:
- `--id/-i <conversation_id>`: Resume or create a conversation with a specific ID
- `--prompt/-p <text>`: Process an initial prompt immediately
- `--directory/-d <path>`: Specify the working directory for Claude Code to run in

Examples:
```bash
# Run with specific working directory
./voice_to_claude_code_fast.py --directory /path/to/project

# Resume conversation in a specific directory
./voice_to_claude_code_fast.py --id abc123 --directory /path/to/project

# Process initial prompt in specific directory
./voice_to_claude_code_fast.py --prompt "list all python files" --directory /path/to/project
```

Press Ctrl+C to exit.
"""

import os
import sys
import json
import yaml
import uuid
import time
import asyncio
import tempfile
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import argparse
import signal
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
from rich.syntax import Syntax
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import pvporcupine
import struct
import logging
import anthropic
import webrtcvad
import collections

# Configuration - default values
TRIGGER_WORDS = ["claude", "cloud", "sonnet", "sonny"]  # List of possible trigger words
STT_MODEL = "small.en"  # Options: tiny.en, base.en, small.en, medium.en, large-v2
PIPER_MODEL = str(Path.home() / ".local/share/piper/models/en_US-amy-low.onnx")  # Full path to Piper model
DEFAULT_CLAUDE_TOOLS = [
    "Bash",
    "Read",
    "Edit",
    "Write",
    "GlobTool",
    "GrepTool",
    "LSTool",
    "Replace",
]

# Sound files
SOUND_WAKEWORD = Path(__file__).parent / "sounds" / "wake-word.mp3"
SOUND_WAKE = Path(__file__).parent / "sounds" / "wake.mp3"
SOUND_TOOL = Path(__file__).parent / "sounds" / "tool.mp3"
SOUND_WAIT = Path(__file__).parent / "sounds" / "wait.mp3"
SOUND_SLEEP = Path(__file__).parent / "sounds" / "sleep.mp3"

# Prompt templates
CLAUDE_PROMPT = """
# Voice-Enabled Claude Code Assistant

You are a helpful assistant that's being used via voice commands. Execute the user's request using your tools.

IMPORTANT: Your responses will be read aloud via text-to-speech. Follow these rules:

1. Keep responses concise and conversational
2. DO NOT use markdown formatting symbols (no *, #, `, [], etc.)
3. Write in plain natural language as if speaking
4. Use words instead of symbols (say "dash" not "-", "at" not "@")

When describing code you've written:
- Just say "I've created/written [brief description]"
- Don't read out the actual code or file paths
- Focus on what it does, not the syntax

When asked to read files, return the entire file content without modification.

{formatted_history}

Now help the user with their latest request.
"""

# Initialize logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("claude_code_assistant")

# Suppress faster_whisper logs
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

console = Console()

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = ["ANTHROPIC_API_KEY"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    console.print(
        f"[bold red]Error: Missing required environment variables: {', '.join(missing_vars)}[/bold red]"
    )
    console.print("Please set these in your .env file or as environment variables.")
    sys.exit(1)


class ClaudeCodeAssistant:
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        working_directory: Optional[str] = None,
    ):
        log.info("Initializing Claude Code Assistant (Fast Mode)")
        self.porcupine = None
        self.whisper_model = None
        self.vad = None
        self.initial_prompt = initial_prompt
        self.working_directory = working_directory
        self.last_interaction_time = None  # Track last interaction time
        self.conversation_timeout = 20 * 60  # 20 minutes in seconds
        self.current_tts_process = None  # Track current TTS process for interruption
        self.current_audio_playback = None  # Track current audio playback
        self.stop_speaking = False  # Track if user wants to stop speaking
        self.sigint_count = 0  # Track number of Ctrl+C presses
        self.last_sigint_time = 0  # Track time of last Ctrl+C

        # Initialize Anthropic client for tool summaries
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        # Check if Piper is available
        self.check_piper()

        # Initialize Porcupine wake word detection
        self.setup_porcupine()

        # Set up conversation ID and history
        if conversation_id:
            # Use the provided ID
            self.conversation_id = conversation_id
        else:
            # Generate a short 5-character ID
            self.conversation_id = "".join(str(uuid.uuid4()).split("-")[0][:5])
        log.info(f"Using conversation ID: {self.conversation_id}")

        # Ensure output directory exists
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Set up the conversation file path
        self.conversation_file = self.output_dir / f"{self.conversation_id}.yml"

        # Load existing conversation or start a new one
        self.conversation_history = self.load_conversation_history()

        # Set up recorder
        self.setup_recorder()

        # Set up SIGINT handler for Ctrl+C
        self.setup_sigint_handler()

    def check_piper(self):
        """Check if Piper is installed and available"""
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            log.info(f"Piper TTS found: {result.stdout.strip()}")
        except FileNotFoundError:
            console.print("[bold red]Error: Piper TTS not found in PATH[/bold red]")
            console.print("Please install Piper: https://github.com/rhasspy/piper")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]Error checking Piper: {e}[/bold red]")
            sys.exit(1)

    def setup_porcupine(self):
        """Initialize Porcupine wake word detection"""
        try:
            # Get Porcupine access key from environment
            access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
            if not access_key:
                console.print("[bold red]Error: PORCUPINE_ACCESS_KEY not found in environment[/bold red]")
                console.print("Please set PORCUPINE_ACCESS_KEY in your .env file")
                console.print("Get a free key from: https://console.picovoice.ai/")
                sys.exit(1)

            # Try to use custom "hey daisy" wake word (Linux version)
            custom_wake_word = "/home/myra/claude-assistant/hey-daisy_en_linux_v3_0_0.ppn"

            if os.path.exists(custom_wake_word):
                try:
                    log.info(f"Attempting to use custom wake word file: {custom_wake_word}")
                    self.porcupine = pvporcupine.create(
                        access_key=access_key,
                        keyword_paths=[custom_wake_word]
                    )
                    self.wake_word_name = "hey daisy"
                    log.info(f"Successfully loaded custom wake word: 'hey daisy'")
                except Exception as custom_error:
                    log.warning(f"Custom wake word failed to load: {custom_error}")
                    log.info("Falling back to built-in 'jarvis' wake word")
                    # Use built-in "jarvis" keyword as fallback (better than "computer")
                    self.porcupine = pvporcupine.create(
                        access_key=access_key,
                        keywords=["jarvis"]
                    )
                    self.wake_word_name = "jarvis"
            else:
                log.warning(f"Custom wake word not found at {custom_wake_word}, using built-in 'jarvis'")
                # Use built-in "jarvis" keyword as fallback (better than "computer")
                # Available keywords: alexa, americano, blueberry, bumblebee, computer, grapefruit,
                # grasshopper, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=["jarvis"]
                )
                self.wake_word_name = "jarvis"

            log.info(f"Porcupine wake word detection initialized with keyword: '{self.wake_word_name}'")
            log.info(f"Audio frame length: {self.porcupine.frame_length}, Sample rate: {self.porcupine.sample_rate}")
        except Exception as e:
            console.print(f"[bold red]Error initializing Porcupine: {e}[/bold red]")
            console.print("Make sure pvporcupine is installed correctly")
            console.print("If using the newer API version, you may need PORCUPINE_ACCESS_KEY")
            sys.exit(1)

    def load_conversation_history(self) -> List[Dict[str, str]]:
        """Load conversation history from YAML file if it exists"""
        if self.conversation_file.exists():
            try:
                log.info(f"Loading existing conversation from {self.conversation_file}")
                with open(self.conversation_file, "r") as f:
                    history = yaml.safe_load(f)
                    if history is None:
                        log.info("Empty conversation file, starting new conversation")
                        return []
                    log.info(f"Loaded {len(history)} conversation turns")
                    return history
            except Exception as e:
                log.error(f"Error loading conversation history: {e}")
                log.info("Starting with empty conversation history")
                return []
        else:
            log.info(
                f"No existing conversation found at {self.conversation_file}, starting new conversation"
            )
            return []

    def save_conversation_history(self) -> None:
        """Save conversation history to YAML file"""
        try:
            log.info(f"Saving conversation history to {self.conversation_file}")
            with open(self.conversation_file, "w") as f:
                yaml.dump(self.conversation_history, f, default_flow_style=False)
            log.info(f"Saved {len(self.conversation_history)} conversation turns")
        except Exception as e:
            log.error(f"Error saving conversation history: {e}")
            console.print(
                f"[bold red]Failed to save conversation history: {e}[/bold red]"
            )

    def check_conversation_timeout(self) -> None:
        """Check if conversation has timed out and create new one if needed"""
        current_time = time.time()

        # If this is the first interaction, just set the time
        if self.last_interaction_time is None:
            self.last_interaction_time = current_time
            return

        # Check if timeout has passed
        time_since_last = current_time - self.last_interaction_time

        if time_since_last > self.conversation_timeout:
            # Only create new conversation if current one has content
            if len(self.conversation_history) > 0:
                log.info(f"Conversation timeout ({time_since_last/60:.1f} minutes). Starting new conversation.")
                console.print(f"[yellow]⏱️  Starting new conversation (inactive for {time_since_last/60:.1f} minutes)[/yellow]")

                # Generate new conversation ID
                self.conversation_id = "".join(str(uuid.uuid4()).split("-")[0][:5])
                self.conversation_file = self.output_dir / f"{self.conversation_id}.yml"

                # Clear conversation history
                self.conversation_history = []

                log.info(f"New conversation ID: {self.conversation_id}")

        # Update last interaction time
        self.last_interaction_time = current_time

    def setup_recorder(self):
        """Set up the faster-whisper model"""
        log.info(f"Setting up STT with faster-whisper model {STT_MODEL}")

        # Initialize faster-whisper model
        self.whisper_model = WhisperModel(
            STT_MODEL,
            device="cpu",  # Use "cuda" if you have GPU
            compute_type="int8"  # Efficient CPU inference
        )

        # Initialize VAD for voice activity detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3 (2 is moderate)

        # Audio recording parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.frame_duration_ms = 30  # VAD frame duration in ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)

        log.info(f"STT initialized with faster-whisper model {STT_MODEL}")

    def setup_sigint_handler(self):
        """Set up SIGINT (Ctrl+C) handler for returning to wake word"""
        def sigint_handler(signum, frame):
            current_time = time.time()

            # If less than 2 seconds since last Ctrl+C, exit
            if current_time - self.last_sigint_time < 2:
                console.print("\n[bold red]Exiting...[/bold red]")
                sys.exit(0)

            # First Ctrl+C returns to wake word
            self.last_sigint_time = current_time
            self.stop_speaking = True
            log.info("Ctrl+C pressed - returning to wake word")
            console.print("\n[bold yellow]⏹  Returning to wake word... (Press Ctrl+C again within 2 seconds to exit)[/bold yellow]")
            self.interrupt_tts()

            # Stop any audio recording
            try:
                sd.stop()
            except:
                pass

        signal.signal(signal.SIGINT, sigint_handler)
        log.info("SIGINT handler set up (Ctrl+C to return to wake word, Ctrl+C twice to exit)")
        console.print("[dim]Press Ctrl+C once to return to wake word, twice quickly to exit[/dim]")

    def format_conversation_history(self) -> str:
        """Format the conversation history in the required format"""
        if not self.conversation_history:
            return ""

        formatted_history = "# Conversation History\n\n"

        for entry in self.conversation_history:
            role = entry["role"].capitalize()
            content = entry["content"]
            formatted_history += f"## {role}\n{content}\n\n"

        return formatted_history

    async def wait_for_wake_word(self) -> bool:
        """Wait for wake word detection using Porcupine"""
        log.info("Listening for wake word...")

        # Reset stop flag when returning to wake word state
        if self.stop_speaking:
            console.print("[yellow]Returning to wake word detection...[/yellow]")
            self.stop_speaking = False

        try:
            # Open audio stream
            audio_stream = sd.InputStream(
                samplerate=self.porcupine.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.porcupine.frame_length
            )

            audio_stream.start()
            console.print(f"[dim]Listening for wake word '{self.wake_word_name}'... (Press Ctrl+C to return to wake word anytime)[/dim]")

            # Play sleep sound when returning to wake word listening
            self.play_sound(SOUND_SLEEP, wait=False)

            while True:
                # Read audio frame
                pcm, overflowed = audio_stream.read(self.porcupine.frame_length)

                if overflowed:
                    log.warning("Audio buffer overflow")

                # Convert to the format Porcupine expects
                pcm = pcm.flatten()

                # Process audio frame
                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    log.info("Wake word detected!")
                    console.print("[bold green]✓ Wake word detected![/bold green]")
                    audio_stream.stop()
                    audio_stream.close()
                    # Play wake word sound without blocking
                    self.play_sound(SOUND_WAKEWORD, wait=False)
                    return True

                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            if audio_stream:
                audio_stream.stop()
                audio_stream.close()
            raise
        except Exception as e:
            log.error(f"Error in wake word detection: {e}")
            if audio_stream:
                audio_stream.stop()
                audio_stream.close()
            return False

    async def listen(self) -> str:
        """Listen for user speech and convert to text using faster-whisper"""
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

        console.print("[dim]Listening... (speak now)[/dim]")

        # Play wake sound and wait for it to finish
        self.play_sound(SOUND_WAKE, wait=True)

        try:
            # Record audio with VAD
            audio_chunks = []
            silence_threshold = 30  # frames of silence before stopping
            silence_count = 0
            speech_started = False

            # Open audio stream
            audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.frame_size
            )

            audio_stream.start()

            # Record until silence is detected after speech
            while True:
                # Check if user pressed Ctrl+C to return to wake word
                if self.stop_speaking:
                    log.info("Ctrl+C detected during listening - aborting")
                    audio_stream.stop()
                    audio_stream.close()
                    return ""

                # Read audio frame
                frame, overflowed = audio_stream.read(self.frame_size)
                if overflowed:
                    log.warning("Audio buffer overflow")

                frame = frame.flatten()
                audio_chunks.append(frame)

                # Convert to bytes for VAD
                frame_bytes = frame.tobytes()

                # Check if frame contains speech
                try:
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                except:
                    # If VAD fails, assume it's speech
                    is_speech = True

                if is_speech:
                    speech_started = True
                    silence_count = 0
                    sys.stdout.write("\r\033[K")  # Clear line
                    sys.stdout.write("Listening: [speaking detected]")
                    sys.stdout.flush()
                elif speech_started:
                    silence_count += 1
                    if silence_count >= silence_threshold:
                        # Silence detected after speech, stop recording
                        break

                # Timeout after 10 seconds
                if len(audio_chunks) > self.sample_rate * 10 / self.frame_size:
                    break

                await asyncio.sleep(0.001)

            audio_stream.stop()
            audio_stream.close()

            if not speech_started:
                console.print("\n[yellow]No speech detected[/yellow]")
                return ""

            # Concatenate audio chunks
            audio_data = np.concatenate(audio_chunks)
            audio_float = audio_data.astype(np.float32) / 32768.0  # Convert to float32

            # Transcribe with faster-whisper
            console.print("\n[dim]Transcribing...[/dim]")
            segments, info = self.whisper_model.transcribe(
                audio_float,
                language="en",
                beam_size=5,
                vad_filter=True
            )

            # Collect transcription
            transcription = " ".join([segment.text for segment in segments]).strip()

            if transcription:
                console.print(
                    Panel(title="You", title_align="left", renderable=Markdown(transcription))
                )
                log.info(f'Heard: "{transcription}"')
                return transcription
            else:
                console.print("\n[yellow]No speech recognized[/yellow]")
                return ""

        except Exception as e:
            log.error(f"Error during speech recognition: {e}")
            console.print(f"\n[red]Error during speech recognition: {e}[/red]")
            return ""

    def play_sound(self, sound_path: Path, wait: bool = True):
        """Play a sound effect using ffmpeg and sounddevice"""
        try:
            if sound_path.exists():
                # Use ffmpeg to decode MP3 to WAV format in memory
                result = subprocess.run(
                    ['ffmpeg', '-i', str(sound_path), '-f', 'wav', '-'],
                    capture_output=True,
                    check=True
                )
                # Read the WAV data
                import io
                audio_data, sample_rate = sf.read(io.BytesIO(result.stdout))
                # Play the sound
                sd.play(audio_data, sample_rate)
                if wait:
                    sd.wait()  # Wait for sound to finish
        except Exception as e:
            log.debug(f"Error playing sound {sound_path}: {e}")

    def interrupt_tts(self):
        """Interrupt current TTS playback"""
        try:
            # Stop audio playback
            sd.stop()

            # Kill TTS process if running
            if self.current_tts_process and self.current_tts_process.poll() is None:
                self.current_tts_process.terminate()
                self.current_tts_process.wait(timeout=1)
        except Exception as e:
            log.debug(f"Error interrupting TTS: {e}")

    async def speak_line(self, text: str, interrupt_previous: bool = True):
        """Convert a single line to speech using Piper TTS with optional interruption"""
        if not text.strip():
            return

        # Check if stop was pressed - if so, skip speaking
        if self.stop_speaking:
            log.info("Stop key was pressed - skipping speech")
            return

        log.info(f'Speaking line: "{text[:50]}..."')

        try:
            # Interrupt previous TTS if requested
            if interrupt_previous:
                self.interrupt_tts()

            # Create temporary files for text and audio
            with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as text_file:
                text_file.write(text)
                text_filename = text_file.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                audio_filename = audio_file.name

            # Run Piper to generate speech
            piper_cmd = [
                "piper",
                "--model", PIPER_MODEL,
                "--output_file", audio_filename
            ]

            with open(text_filename, 'r') as text_input:
                self.current_tts_process = subprocess.Popen(
                    piper_cmd,
                    stdin=text_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Wait for TTS generation (check for stop key periodically)
                timeout = 30
                poll_interval = 0.1
                elapsed = 0
                while elapsed < timeout:
                    if self.current_tts_process.poll() is not None:
                        break
                    if self.stop_speaking:
                        log.info("Stop key pressed during TTS generation")
                        self.interrupt_tts()
                        os.unlink(text_filename)
                        if os.path.exists(audio_filename):
                            os.unlink(audio_filename)
                        return
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

            if self.current_tts_process.returncode != 0:
                stderr = self.current_tts_process.stderr.read() if self.current_tts_process.stderr else ""
                raise Exception(f"Piper failed: {stderr}")

            # Play audio (check for ESC during playback)
            data, samplerate = sf.read(audio_filename)
            sd.play(data, samplerate)

            # Wait for audio to finish (but can be interrupted by stop key)
            # Poll more frequently to catch stop key quickly
            while True:
                if self.stop_speaking:
                    log.info("Stop key pressed during audio playback")
                    console.print("\n[yellow]⏹  Speech interrupted[/yellow]")
                    sd.stop()
                    break
                # Check if audio is still playing
                try:
                    if not sd.get_stream().active:
                        break
                except:
                    # If checking stream fails, just break
                    break
                await asyncio.sleep(0.05)

            # Clean up temporary files
            os.unlink(text_filename)
            os.unlink(audio_filename)

        except Exception as e:
            log.error(f"Error in speech synthesis: {str(e)}")
            # Don't print error to console to avoid clutter in streaming mode

    async def speak(self, text: str):
        """Convert text to speech using Piper TTS (legacy method for backward compatibility)"""
        await self.speak_line(text, interrupt_previous=False)

    async def summarize_tool_use(self, tool_name: str, tool_input: dict) -> str:
        """Use Claude Haiku to summarize what a tool is doing"""
        try:
            # Create a concise prompt for summarization
            prompt = f"""Summarize this action in one SHORT, SPECIFIC sentence (under 12 words) using present continuous tense (verb + -ing).

Tool: {tool_name}
Input: {json.dumps(tool_input, indent=2)}

Be SPECIFIC - include important details like:
- File/directory names or patterns
- Search terms or paths
- Key parameters

Examples:
- "Searching home folder for Python files"
- "Reading README.md file"
- "Listing contents of Photos directory"
- "Running git status in current repo"

Reply with ONLY the specific summary sentence starting with a verb ending in -ing, no extra words."""

            # Call Claude Haiku for fast summary
            message = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )

            summary = message.content[0].text.strip()
            return summary

        except Exception as e:
            log.error(f"Error summarizing tool use: {e}")
            # Fallback to simple message
            return f"Using {tool_name}"

    async def process_message(self, message: str) -> Optional[str]:
        """Process the user message and run Claude Code with real-time streaming"""
        log.info(f'Processing message: "{message}"')

        # Wake word detection is now handled separately, so we always process the message
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Prepare the prompt for Claude Code including conversation history
        formatted_history = self.format_conversation_history()
        prompt = CLAUDE_PROMPT.format(formatted_history=formatted_history)

        # Execute Claude Code as a subprocess with streaming
        log.info("Starting Claude Code subprocess with real-time streaming...")
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--allowedTools",
        ] + DEFAULT_CLAUDE_TOOLS

        console.print("\n[bold blue]🔄 Running Claude Code (streaming)...[/bold blue]\n")

        try:
            # Use Popen for real-time streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=self.working_directory
            )

            full_response = []
            accumulated_text = []  # Track text blocks for conversation history

            # Read output line by line in real-time (each line is a JSON object)
            for line in process.stdout:
                if not line.strip():
                    continue

                try:
                    # Parse JSON streaming output
                    event = json.loads(line.strip())
                    event_type = event.get("type")

                    # Handle assistant messages
                    if event_type == "assistant":
                        message = event.get("message", {})
                        content = message.get("content", [])

                        for item in content:
                            item_type = item.get("type")

                            if item_type == "text":
                                # Text content from Claude
                                text = item.get("text", "")
                                if text:
                                    accumulated_text.append(text)

                                    # Speak complete sentences and display them
                                    if text.endswith((".", "!", "?", "\n")):
                                        full_text = "".join(accumulated_text)
                                        # Display the sentence in a panel with blue border
                                        console.print(Panel(
                                            full_text.strip(),
                                            title="Assistant",
                                            title_align="left",
                                            border_style="blue"
                                        ))
                                        await self.speak_line(full_text.strip(), interrupt_previous=True)
                                        accumulated_text = []

                            elif item_type == "tool_use":
                                # Tool usage notification with intelligent summary
                                tool_name = item.get("name", "unknown")
                                tool_input = item.get("input", {})

                                # Play tool sound without blocking
                                self.play_sound(SOUND_TOOL, wait=False)

                                # Format tool message with key details
                                if tool_name == "Bash" and "command" in tool_input:
                                    tool_msg = f"[Using {tool_name} ({tool_input['command']})]"
                                elif "file_path" in tool_input:
                                    tool_msg = f"[Using {tool_name} ({tool_input['file_path']})]"
                                elif "pattern" in tool_input:
                                    tool_msg = f"[Using {tool_name} ({tool_input['pattern']})]"
                                else:
                                    tool_msg = f"[Using {tool_name}]"

                                console.print(f"\n[dim cyan]{tool_msg}[/dim cyan]")

                                # Get AI summary of what the tool is doing
                                summary = await self.summarize_tool_use(tool_name, tool_input)
                                await self.speak_line(summary, interrupt_previous=True)

                    elif event_type == "user":
                        # Tool results - usually we don't speak these
                        message = event.get("message", {})
                        content = message.get("content", [])

                        for item in content:
                            if item.get("type") == "tool_result":
                                tool_msg = "[Tool completed]"
                                console.print(f"[dim green]{tool_msg}[/dim green]")

                    # Collect all events for debugging
                    full_response.append(event)

                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
                    # Fallback: treat as plain text
                    console.print(line.rstrip())
                    await self.speak_line(line.strip(), interrupt_previous=True)

            # Speak any remaining accumulated text
            if accumulated_text:
                full_text = "".join(accumulated_text)
                await self.speak_line(full_text.strip(), interrupt_previous=True)

            # Wait for process to complete
            process.wait()

            # Check for errors
            if process.returncode != 0:
                stderr = process.stderr.read()
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)

            # Extract text content from JSON events for conversation history
            response_text = []
            for event in full_response:
                if isinstance(event, dict) and event.get("type") == "assistant":
                    message = event.get("message", {})
                    content = message.get("content", [])
                    for item in content:
                        if item.get("type") == "text":
                            response_text.append(item.get("text", ""))

            response = "".join(response_text)

            log.info(f"Claude Code succeeded, output length: {len(response)}")

            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            # Save the updated conversation history
            self.save_conversation_history()

            return response

        except subprocess.CalledProcessError as e:
            error_msg = f"Claude Code failed with exit code: {e.returncode}"
            log.error(f"{error_msg}\nError: {e.stderr if hasattr(e, 'stderr') else 'Unknown'}...")

            error_response = "I'm sorry, but I encountered an error while processing your request. Please try again."

            # Speak the error
            await self.speak_line(error_response, interrupt_previous=True)

            self.conversation_history.append(
                {"role": "assistant", "content": error_response}
            )

            # Save the updated conversation history even when there's an error
            self.save_conversation_history()

            return error_response
        except Exception as e:
            log.error(f"Unexpected error in process_message: {e}")
            error_response = "I encountered an unexpected error. Please try again."
            await self.speak_line(error_response, interrupt_previous=True)
            return error_response

    async def conversation_loop(self):
        """Run the main conversation loop"""
        log.info("Starting conversation loop")

        console.print(
            Panel.fit(
                "[bold magenta]🎤 Claude Code Voice Assistant Ready (Fast Mode with Wake Word)[/bold magenta]\n"
                f"Say '{self.wake_word_name}' to activate, then speak your request.\n"
                f"The assistant will process with Claude Code and respond using Piper TTS (model: {PIPER_MODEL}).\n"
                f"STT model: {STT_MODEL}\n"
                f"Conversation ID: {self.conversation_id}\n"
                f"Saving conversation to: {self.conversation_file}\n"
                f"[yellow]Press Ctrl+C once to return to wake word (from any state), twice quickly to exit[/yellow]"
            )
        )

        try:
            while True:
                # First, wait for wake word
                wake_word_detected = await self.wait_for_wake_word()

                if not wake_word_detected:
                    console.print("[yellow]Wake word detection failed. Retrying...[/yellow]")
                    continue

                # Check if conversation has timed out
                self.check_conversation_timeout()

                # Wake word detected, now listen for the actual command
                user_text = await self.listen()

                if not user_text:
                    console.print("[yellow]No speech detected. Try again.[/yellow]")
                    continue

                # Process the message (now without checking for trigger words since wake word was already detected)
                # Note: process_message now handles speaking each line in real-time
                response = await self.process_message(user_text)

                # Check if user interrupted speech - if so, skip to wake word immediately
                if self.stop_speaking:
                    log.info("Speech was interrupted - returning to wake word detection")
                    continue

                # No need to speak the response again since we already spoke each line
                # Just give a small break between interactions
                if response:
                    await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping assistant...[/bold red]")
            log.info("Conversation loop stopped by keyboard interrupt")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            log.error(f"Error in conversation loop: {str(e)}", exc_info=True)
        finally:
            # Safe cleanup
            try:
                if hasattr(self, "porcupine") and self.porcupine:
                    # Shutdown Porcupine properly
                    self.porcupine.delete()
                # Stop any audio
                sd.stop()
            except Exception as shutdown_error:
                log.error(f"Error during shutdown: {str(shutdown_error)}")

            console.print("[bold red]Assistant stopped.[/bold red]")
            log.info("Conversation loop ended")


async def main():
    """Main entry point for the assistant"""
    log.info("Starting Claude Code Voice Assistant (Fast Mode)")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fast voice-enabled Claude Code assistant with Piper TTS")
    parser.add_argument(
        "--id",
        "-i",
        type=str,
        help="Unique ID for the conversation. If provided and exists, will load existing conversation.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Initial prompt to process immediately (will be prefixed with trigger word)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        help="Working directory for Claude Code to run in (defaults to current directory)",
    )
    args = parser.parse_args()

    # Validate directory if provided
    if args.directory:
        dir_path = Path(args.directory)
        if not dir_path.exists():
            console.print(f"[bold red]Error: Directory does not exist: {args.directory}[/bold red]")
            sys.exit(1)
        if not dir_path.is_dir():
            console.print(f"[bold red]Error: Path is not a directory: {args.directory}[/bold red]")
            sys.exit(1)
        log.info(f"Using working directory: {args.directory}")

    # Create assistant instance with conversation ID, initial prompt, and working directory
    assistant = ClaudeCodeAssistant(
        conversation_id=args.id,
        initial_prompt=args.prompt,
        working_directory=args.directory
    )

    # Show some helpful information about the conversation
    if args.id:
        if assistant.conversation_file.exists():
            log.info(f"Resuming existing conversation with ID: {args.id}")
            console.print(
                f"[bold green]Resuming conversation {args.id} with {len(assistant.conversation_history)} turns[/bold green]"
            )
        else:
            log.info(f"Starting new conversation with user-provided ID: {args.id}")
            console.print(
                f"[bold blue]Starting new conversation with ID: {args.id}[/bold blue]"
            )
    else:
        log.info(
            f"Starting new conversation with auto-generated ID: {assistant.conversation_id}"
        )
        console.print(
            f"[bold blue]Starting new conversation with auto-generated ID: {assistant.conversation_id}[/bold blue]"
        )

    log.info(f"Conversation will be saved to: {assistant.conversation_file}")
    console.print(f"[bold]Conversation file: {assistant.conversation_file}[/bold]")

    # Process initial prompt if provided
    if args.prompt:
        log.info(f"Processing initial prompt: {args.prompt}")
        console.print(
            f"[bold cyan]Processing initial prompt: {args.prompt}[/bold cyan]"
        )

        # Create a full prompt that includes the trigger word to ensure it's processed
        initial_prompt = f"{TRIGGER_WORDS[0]} {args.prompt}"

        # Process the initial prompt
        # Note: process_message now handles speaking each line in real-time
        response = await assistant.process_message(initial_prompt)

        # No need to speak the response again since we already spoke each line

    # Run the conversation loop
    await assistant.conversation_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Program terminated by user")
        console.print("\n[bold red]Program terminated by user.[/bold red]")
