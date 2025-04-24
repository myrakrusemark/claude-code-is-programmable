#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "python-dotenv",
#   "rich"
# ]
# ///

import os
import sys
import subprocess
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.syntax import Syntax
from rich import print as rprint

# Initialize rich console
console = Console()

# Load environment variables for Notion API
load_dotenv()
NOTION_API_SECRET = os.getenv("NOTION_INTERNAL_INTEGRATION_SECRET")
if not NOTION_API_SECRET:
    console.print(
        "[bold red]ERROR: NOTION_INTERNAL_INTEGRATION_SECRET not found in environment variables.[/bold red]"
    )
    sys.exit(1)

# Check for the page name argument
if len(sys.argv) < 2:
    console.print(
        "[bold red]ERROR: Please provide a Notion page name as an argument.[/bold red]"
    )
    console.print("Usage: uv run claude_code_is_programmable_3.py <notion_page_name>")
    sys.exit(1)

page_name = sys.argv[1]

# Define the allowed tools for Claude
allowed_tools = [
    # Standard Claude Code tools
    "Bash",
    "Edit",
    "View",
    "GlobTool",
    "GrepTool",
    "LSTool",
    "BatchTool",
    "AgentTool",
    "WebFetchTool",
    "Write",
    # Notion API tools
    "mcp__notionApi__API-get-user",
    "mcp__notionApi__API-get-users",
    "mcp__notionApi__API-get-self",
    "mcp__notionApi__API-post-database-query",
    "mcp__notionApi__API-post-search",
    "mcp__notionApi__API-get-block-children",
    "mcp__notionApi__API-patch-block-children",
    "mcp__notionApi__API-retrieve-a-block",
    "mcp__notionApi__API-update-a-block",
    "mcp__notionApi__API-delete-a-block",
    "mcp__notionApi__API-retrieve-a-page",
    "mcp__notionApi__API-patch-page",
    "mcp__notionApi__API-post-page",
    "mcp__notionApi__API-create-a-database",
    "mcp__notionApi__API-update-a-database",
    "mcp__notionApi__API-retrieve-a-database",
    "mcp__notionApi__API-retrieve-a-page-property",
    "mcp__notionApi__API-retrieve-a-comment",
    "mcp__notionApi__API-create-a-comment",
]

# Create the prompt for Claude
prompt = f"""
# Notion Todo Code Generation Agent

## Objective
You are an agent that will:
1. Find and read a Notion page named "{page_name}"
2. Extract all todo items from the page
3. For each incomplete todo, implement the code changes described in the todo
4. Commit the changes with a descriptive message
5. Mark the todo item as complete in Notion
6. Continue to the next todo item

## Process - Follow these steps exactly:

### Step 1: Find the Notion page
- Use the Notion API via the mcp__notionApi__API-post-search tool to search for a page with the name "{page_name}"
- Extract the page ID from the search results

### Step 2: Get page content
- Use the mcp__notionApi__API-retrieve-a-page tool to get the page details
- Use the mcp__notionApi__API-get-block-children tool to get the page blocks
- Look for any to_do blocks, which represent your todo items
- For each to_do block, capture:
  - The block ID
  - The content text
  - Whether it's already checked/completed

### Step 3: Process each todo
For each UNCHECKED todo item:
1. Read and understand the todo description
2. Implement the code changes described:
   - Use GlobTool, GrepTool, View to explore the codebase
   - Use Edit or Replace to modify or create files
   - Use Bash when necessary to run commands
3. Test your implementation if tests are available
4. Stage and commit your changes with a descriptive message:
   ```bash
   git add .
   git commit -m "Descriptive message about what was implemented"
   ```
5. Mark the todo as complete in Notion using the mcp__notionApi__API-update-a-block tool

### Step 4: Wrap up
- Provide a summary of all todos processed and changes made

## Important Notes:
- Skip any todos that are already checked/complete
- Process todos in the order they appear on the page
- Make one commit per todo item
- Ensure each commit message clearly describes what was implemented
- If a todo cannot be completed, note why but don't mark it as complete
- If a todo is already completed, skip it

## Available Notion Tools:
You have access to the standard Claude Code tools like Bash, Edit, Replace, View, etc., as well as the complete set of Notion API tools:
- mcp__notionApi__API-post-search: Use this to find the Notion page by name
- mcp__notionApi__API-get-block-children: Use this to retrieve the todo items from the page
- mcp__notionApi__API-update-a-block: Use this to mark todos as complete
And many other Notion API tools as needed.

Now begin your task by finding the Notion page named "{page_name}" and processing its todos.
"""

# Execute the Claude command with stream-json output format
try:
    console.print(
        f"[bold blue]🤖 Starting Claude Code to process todos from Notion page:[/bold blue] [yellow]{page_name}[/yellow]"
    )

    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--allowedTools",
    ] + allowed_tools

    # Start the process and read output as it comes
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Process and display JSON output in real-time
    console.print("\n[bold green]📊 Streaming Claude output:[/bold green]")
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        syntax = Syntax(line, "json", theme="monokai", line_numbers=False)
        console.print(syntax)

    # Check for any errors
    stderr = process.stderr.read()
    if stderr:
        console.print(f"[bold red]⚠️ Error output from Claude:[/bold red]\n{stderr}")

    # Get return code
    return_code = process.wait()
    if return_code == 0:
        console.print(f"[bold green]✅ Claude Code completed successfully[/bold green]")
    else:
        console.print(
            f"[bold red]❌ Claude Code failed with exit code: {return_code}[/bold red]"
        )
        sys.exit(return_code)

except subprocess.CalledProcessError as e:
    console.print(f"[bold red]❌ Error executing Claude Code: {str(e)}[/bold red]")
    sys.exit(1)
except Exception as e:
    console.print(f"[bold red]❌ Unexpected error: {str(e)}[/bold red]")
    sys.exit(1)
