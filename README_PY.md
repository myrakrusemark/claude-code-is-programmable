# Python Files in Claude Code is Programmable

This document provides an overview of the Python files in this repository.

## Main Scripts

### `claude_code_is_programmable_2.py`
A simple script that uses Claude Code to create a zero-library CLI todo app. It launches Claude with a specific prompt to create a TypeScript todo app, then commits the changes to git.

### `claude_code_is_programmable_3.py`
A more advanced script that integrates Claude Code with Notion. It finds a Notion page, extracts todo items, and processes them one by one. For each todo, it implements the required code changes, commits them, and marks the todo as complete in Notion.

### `claude_code_is_programmable_4.py`
Demonstrates different output formats for Claude Code. This script shows how to use the `--output-format` flag with options like text, JSON, and streaming JSON to control how Claude's responses are formatted.

### `aider_is_programmable_2.py`
Similar to claude_code_is_programmable_2.py but uses the Aider tool instead of Claude Code directly. It creates and checks out a git branch, runs Aider to create a TypeScript todo app, and commits the changes.

## Utility Scripts

### `anthropic_search.py`
A command-line tool for searching the web using Anthropic's Claude API. It supports features like domain filtering, location-based search, and formatting of search results with citations.

### `voice_to_claude_code.py`
A voice-enabled Claude Code assistant that allows for interaction with Claude Code using speech. It combines RealtimeSTT for speech recognition and OpenAI TTS for speech output, with support for trigger word activation and conversation history tracking.

## Bonus Scripts

### `bonus/starter_notion_agent.py`
A starter script for creating an agent with access to the Notion API using the OpenAI Agents framework. It demonstrates setting up an MCP server for Notion and running a simple test query.

### `bonus/claude_code_inside_openai_agent_sdk_4_bonus.py`
An advanced implementation that integrates Claude Code with the OpenAI Agents framework. It creates a Notion Code Generator agent that can find a Notion page, get its content, process todos, write code based on each todo, and mark todos as complete.