# Claude Code is Programmable

This repository demonstrates how to use Claude Code programmatically, showcasing examples in different programming languages.

## File Descriptions

### Shell Scripts
- `claude_code_is_programmable_1.sh`: Simple shell script that uses Claude Code's CLI to generate a basic "hello.js" script with limited allowed tools.
- `aider_is_programmable_1.sh`: Similar script using Aider to create a "hello.js" file.
- `reset.sh`: Utility script to clean up branches and directories created by the demo scripts.

### Python Files
- `claude_code_is_programmable_2.py`: Python script that executes Claude Code to create a TypeScript CLI todo app, with permissions for Edit, Replace, Bash, and Create tools.
- `claude_code_is_programmable_3.py`: Advanced Python script integrating Claude Code with Notion API for todo management, including rich console output and streaming results.
- `aider_is_programmable_2.py`: Python script that uses Aider to create a TypeScript todo application with git operations.

### JavaScript Files
- `claude_code_is_programmable_2.js`: JavaScript version of the Claude Code script that creates a TypeScript todo app, with permissions for Edit, Replace, Bash, and Create tools.
- `aider_is_programmable_2.js`: JavaScript version of the Aider script for creating a TypeScript todo app with git operations.

### Bonus Directory
- `starter_notion_agent.py`: A starter template for creating a Notion agent using the OpenAI Agent SDK.
- `claude_code_inside_openai_agent_sdk_4_bonus.py`: An advanced implementation that integrates Claude Code within the OpenAI Agent SDK, demonstrating how to process Notion todos with an agent-based architecture.

## Core Tools Available in Claude Code

- Task: Launch an agent to perform complex tasks
- Bash: Execute bash commands in a shell
- Batch: Run multiple tools in parallel
- Glob: Find files matching patterns
- Grep: Search file contents with regex
- LS: List directory contents
- Read: Read file contents
- Edit: Make targeted edits to files
- Write: Create or overwrite files
- NotebookRead/Edit: Work with Jupyter notebooks
- WebFetch: Get content from websites