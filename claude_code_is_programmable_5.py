#!/usr/bin/env -S uv run --script

# Summary: Script that runs Claude Code CLI with two prompts: one to read all Python files
# and another to add summaries to each Python file, demonstrating Claude Code's programmability.

import subprocess

# First prompt: Read all Python files
read_prompt = """
READ all .PY files in PARALLEL
"""

# Second prompt: Add summaries to all Python files
write_prompt = """
WRITE a small summary at the top of each of those files in parallel
"""

# Function to run Claude command with a given prompt
def run_claude_command(prompt):
    command = ["claude", "-p", prompt, "--allowedTools", "Edit", "Bash", "Write", "Read", "MultiEdit", "Batch"]
    
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    
    return process.stdout

# Execute the read prompt
print("Running read prompt...")
read_output = run_claude_command(read_prompt)
print(f"Read operation completed successfully.\n")

# Execute the write prompt
print("Running write prompt...")
write_output = run_claude_command(write_prompt)
print(f"Write operation completed successfully.\n")

print("All operations completed successfully.")