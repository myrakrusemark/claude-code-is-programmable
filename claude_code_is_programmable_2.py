#!/usr/bin/env -S uv run --script

import subprocess

prompt = """

GIT checkout a NEW branch.

CREATE ./cc_todo/todo.ts: a zero library CLI todo app with basic CRUD. 

THEN GIT stage, commit and SWITCH back to main.

"""

command = ["claude", "-p", prompt, "--allowedTools", "Edit", "Bash", "Write"]

# Capture Claude's output so we can display it
process = subprocess.run(
    command,
    check=True,
    capture_output=True,
    text=True,
)

print(f"Claude process exited with output: {process.stdout}")
