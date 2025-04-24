#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

import subprocess
import os

# Create directory if it doesn't exist
todo_dir = "./cc_todo"
os.makedirs(todo_dir, exist_ok=True)
todo_file = f"{todo_dir}/todo.ts"

# Generate a random branch name
branch_name = f"feature-todo-app"

try:
    # 1. Create and checkout a new branch
    print(f"Creating and checking out new branch: {branch_name}")
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)

    # 2. Run aider directly with the todo task
    print("Running aider to create todo app...")
    aider_cmd = [
        "aider",
        "--no-git",  # We'll handle git ourselves
        todo_file,
        "--message",
        "CREATE ./cc_todo/todo.ts: a zero library CLI todo app with basic CRUD.",
    ]
    subprocess.run(aider_cmd, check=True)

    # 3. Git operations - stage and commit
    print("Staging and committing changes...")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add TypeScript todo app with CRUD functionality"],
        check=True,
    )

    # 4. Switch back to main branch
    print("Switching back to main branch...")
    subprocess.run(["git", "checkout", "main"], check=True)

    print(f"Task completed. Changes committed to branch: {branch_name}")

except subprocess.CalledProcessError as e:
    print(f"Command failed: {e}")
except Exception as e:
    print(f"Error: {e}")
    # Try to return to main branch if something went wrong
    try:
        subprocess.run(["git", "checkout", "main"], check=True)
    except:
        pass
