# Claude Code Tutorials

This document provides step-by-step tutorials for common workflows with Claude Code, extracted from [Anthropic's official documentation](https://docs.anthropic.com/en/docs/claude-code/tutorials).

## Table of contents

- [Resume previous conversations](#resume-previous-conversations)
- [Understand new codebases](#understand-new-codebases)
- [Fix bugs efficiently](#fix-bugs-efficiently)
- [Refactor code](#refactor-code)
- [Work with tests](#work-with-tests)
- [Create pull requests](#create-pull-requests)
- [Handle documentation](#handle-documentation)
- [Work with images](#work-with-images)
- [Use extended thinking](#use-extended-thinking)
- [Set up project memory](#set-up-project-memory)
- [Set up Model Context Protocol (MCP)](#set-up-model-context-protocol-mcp)
- [Use Claude as a unix-style utility](#use-claude-as-a-unix-style-utility)
- [Create custom slash commands](#create-custom-slash-commands)
- [Run parallel Claude Code sessions with Git worktrees](#run-parallel-claude-code-sessions-with-git-worktrees)

## Resume previous conversations

### Continue your work seamlessly

**When to use:** You've been working on a task with Claude Code and need to continue where you left off in a later session.

Claude Code provides two options for resuming previous conversations:

- `--continue` to automatically continue the most recent conversation
- `--resume` to display a conversation picker

1. Continue the most recent conversation:
```bash
claude --continue
```

2. Continue in non-interactive mode:
```bash
claude --continue --print "Continue with my task"
```

3. Show conversation picker:
```bash
claude --resume
```

**Tips:**
- Conversation history is stored locally on your machine
- Use `--continue` for quick access to your most recent conversation
- Use `--resume` when you need to select a specific past conversation
- When resuming, you'll see the entire conversation history before continuing
- The resumed conversation starts with the same model and configuration as the original

## Understand new codebases

### Get a quick codebase overview

**When to use:** You've just joined a new project and need to understand its structure quickly.

1. Navigate to the project root directory:
```bash
cd /path/to/project
```

2. Start Claude Code:
```bash
claude
```

3. Ask for a high-level overview:
```
> give me an overview of this codebase
```

4. Dive deeper into specific components:
```
> explain the main architecture patterns used here
> what are the key data models?
> how is authentication handled?
```

**Tips:**
- Start with broad questions, then narrow down to specific areas
- Ask about coding conventions and patterns used in the project
- Request a glossary of project-specific terms

### Find relevant code

**When to use:** You need to locate code related to a specific feature or functionality.

1. Ask Claude to find relevant files:
```
> find the files that handle user authentication
```

2. Get context on how components interact:
```
> how do these authentication files work together?
```

3. Understand the execution flow:
```
> trace the login process from front-end to database
```

## Fix bugs efficiently

### Diagnose error messages

**When to use:** You've encountered an error message and need to find and fix its source.

1. Share the error with Claude:
```
> I'm seeing an error when I run npm test
```

2. Ask for fix recommendations:
```
> suggest a few ways to fix the @ts-ignore in user.ts
```

3. Apply the fix:
```
> update user.ts to add the null check you suggested
```

**Tips:**
- Tell Claude the command to reproduce the issue and get a stack trace
- Mention any steps to reproduce the error
- Let Claude know if the error is intermittent or consistent

## Refactor code

### Modernize legacy code

**When to use:** You need to update old code to use modern patterns and practices.

1. Identify legacy code for refactoring:
```
> find deprecated API usage in our codebase
```

2. Get refactoring recommendations:
```
> suggest how to refactor utils.js to use modern JavaScript features
```

3. Apply the changes safely:
```
> refactor utils.js to use ES2024 features while maintaining the same behavior
```

4. Verify the refactoring:
```
> run tests for the refactored code
```

**Tips:**
- Ask Claude to explain the benefits of the modern approach
- Request that changes maintain backward compatibility when needed
- Do refactoring in small, testable increments

## Work with tests

### Add test coverage

**When to use:** You need to add tests for uncovered code.

1. Identify untested code:
```
> find functions in NotificationsService.swift that are not covered by tests
```

2. Generate test scaffolding:
```
> add tests for the notification service
```

3. Add meaningful test cases:
```
> add test cases for edge conditions in the notification service
```

4. Run and verify tests:
```
> run the new tests and fix any failures
```

**Tips:**
- Ask for tests that cover edge cases and error conditions
- Request both unit and integration tests when appropriate
- Have Claude explain the testing strategy

## Create pull requests

### Generate comprehensive PRs

**When to use:** You need to create a well-documented pull request for your changes.

1. Summarize your changes:
```
> summarize the changes I've made to the authentication module
```

2. Generate a PR with Claude:
```
> create a pr
```

3. Review and refine:
```
> enhance the PR description with more context about the security improvements
```

4. Add testing details:
```
> add information about how these changes were tested
```

**Tips:**
- Ask Claude directly to make a PR for you
- Review Claude's generated PR before submitting
- Ask Claude to highlight potential risks or considerations

## Handle documentation

### Generate code documentation

**When to use:** You need to add or update documentation for your code.

1. Identify undocumented code:
```
> find functions without proper JSDoc comments in the auth module
```

2. Generate documentation:
```
> add JSDoc comments to the undocumented functions in auth.js
```

3. Review and enhance:
```
> improve the generated documentation with more context and examples
```

4. Verify documentation:
```
> check if the documentation follows our project standards
```

**Tips:**
- Specify the documentation style you want (JSDoc, docstrings, etc.)
- Ask for examples in the documentation
- Request documentation for public APIs, interfaces, and complex logic

## Work with images

### Analyze images and screenshots

**When to use:** You need to work with images in your codebase or get Claude's help analyzing image content.

1. Add an image to the conversation using one of these methods:
   - Drag and drop an image into the Claude Code window
   - Copy an image and paste it into the CLI with cmd+v (on Mac)
   - Provide an image path: `claude "Analyze this image: /path/to/your/image.png"`

2. Ask Claude to analyze the image:
```
> What does this image show?
> Describe the UI elements in this screenshot
> Are there any problematic elements in this diagram?
```

3. Use images for context:
```
> Here's a screenshot of the error. What's causing it?
> This is our current database schema. How should we modify it for the new feature?
```

4. Get code suggestions from visual content:
```
> Generate CSS to match this design mockup
> What HTML structure would recreate this component?
```

**Tips:**
- Use images when text descriptions would be unclear or cumbersome
- Include screenshots of errors, UI designs, or diagrams for better context
- You can work with multiple images in a conversation
- Image analysis works with diagrams, screenshots, mockups, and more

## Use extended thinking

### Leverage Claude's extended thinking for complex tasks

**When to use:** When working on complex architectural decisions, challenging bugs, or planning multi-step implementations that require deep reasoning.

1. Provide context and ask Claude to think:
```
> I need to implement a new authentication system using OAuth2 for our API. Think deeply about the best approach for implementing this in our codebase.
```

2. Refine the thinking with follow-up prompts:
```
> think about potential security vulnerabilities in this approach
> think harder about edge cases we should handle
```

**Tips to get the most value out of extended thinking:**

Extended thinking is most valuable for complex tasks such as:
- Planning complex architectural changes
- Debugging intricate issues
- Creating implementation plans for new features
- Understanding complex codebases
- Evaluating tradeoffs between different approaches

The way you prompt for thinking results in varying levels of thinking depth:
- "think" triggers basic extended thinking
- intensifying phrases such as "think more", "think a lot", "think harder", or "think longer" triggers deeper thinking

Claude will display its thinking process as italic gray text above the response.

## Set up project memory

### Create an effective CLAUDE.md file

**When to use:** You want to set up a CLAUDE.md file to store important project information, conventions, and frequently used commands.

1. Bootstrap a CLAUDE.md for your codebase:
```
> /init
```

**Tips:**
- Include frequently used commands (build, test, lint) to avoid repeated searches
- Document code style preferences and naming conventions
- Add important architectural patterns specific to your project
- CLAUDE.md memories can be used for both instructions shared with your team and for your individual preferences

## Set up Model Context Protocol (MCP)

Model Context Protocol (MCP) is an open protocol that enables LLMs to access external tools and data sources.

### Configure MCP servers

**When to use:** You want to enhance Claude's capabilities by connecting it to specialized tools and external servers using the Model Context Protocol.

1. Add an MCP Stdio Server:
```bash
# Basic syntax
claude mcp add <n> <command> [args...]

# Example: Adding a local server
claude mcp add my-server -e API_KEY=123 -- /path/to/server arg1 arg2
```

2. Add an MCP SSE Server:
```bash
# Basic syntax
claude mcp add --transport sse <n> <url>

# Example: Adding an SSE server
claude mcp add --transport sse sse-server https://example.com/sse-endpoint
```

3. Manage your MCP servers:
```bash
# List all configured servers
claude mcp list

# Get details for a specific server
claude mcp get my-server

# Remove a server
claude mcp remove my-server
```

**Tips:**
- Use the `-s` or `--scope` flag to specify where the configuration is stored:
  - `local` (default): Available only to you in the current project
  - `project`: Shared with everyone in the project via `.mcp.json` file
  - `user`: Available to you across all projects
- Set environment variables with `-e` or `--env` flags (e.g., `-e KEY=value`)
- Configure MCP server startup timeout using the MCP_TIMEOUT environment variable
- Check MCP server status any time using the `/mcp` command within Claude Code

### Understanding MCP server scopes

**When to use:** You want to understand how different MCP scopes work and how to share servers with your team.

1. Local-scoped MCP servers:
```bash
# Add a local-scoped server (default)
claude mcp add my-private-server /path/to/server

# Explicitly specify local scope
claude mcp add my-private-server -s local /path/to/server
```

2. Project-scoped MCP servers (.mcp.json):
```bash
# Add a project-scoped server
claude mcp add shared-server -s project /path/to/server
```

3. User-scoped MCP servers:
```bash
# Add a user server
claude mcp add my-user-server -s user /path/to/server
```

**Tips:**
- Local-scoped servers take precedence over project-scoped and user-scoped servers with the same name
- Project-scoped servers (in `.mcp.json`) take precedence over user-scoped servers with the same name
- Before using project-scoped servers from `.mcp.json`, Claude Code will prompt you to approve them for security
- The `.mcp.json` file is intended to be checked into version control to share MCP servers with your team

## Use Claude as a unix-style utility

### Add Claude to your verification process

**When to use:** You want to use Claude Code as a linter or code reviewer.

Add Claude to your build script:
```json
// package.json
{
    ...
    "scripts": {
        ...
        "lint:claude": "claude -p 'you are a linter. please look at the changes vs. main and report any issues related to typos. report the filename and line number on one line, and a description of the issue on the second line. do not return any other text.'"
    }
}
```

### Pipe in, pipe out

**When to use:** You want to pipe data into Claude, and get back data in a structured format.

Pipe data through Claude:
```bash
cat build-error.txt | claude -p 'concisely explain the root cause of this build error' > output.txt
```

### Control output format

**When to use:** You need Claude's output in a specific format, especially when integrating Claude Code into scripts or other tools.

1. Use text format (default):
```bash
cat data.txt | claude -p 'summarize this data' --output-format text > summary.txt
```

2. Use JSON format:
```bash
cat code.py | claude -p 'analyze this code for bugs' --output-format json > analysis.json
```

3. Use streaming JSON format:
```bash
cat log.txt | claude -p 'parse this log file for errors' --output-format stream-json
```

**Tips:**
- Use `--output-format text` for simple integrations where you just need Claude's response
- Use `--output-format json` when you need the full conversation log
- Use `--output-format stream-json` for real-time output of each conversation turn

## Create custom slash commands

Claude Code supports custom slash commands that you can create to quickly execute specific prompts or tasks.

### Create project-specific commands

**When to use:** You want to create reusable slash commands for your project that all team members can use.

1. Create a commands directory in your project:
```bash
mkdir -p .claude/commands
```

2. Create a Markdown file for each command:
```bash
echo "Analyze the performance of this code and suggest three specific optimizations:" > .claude/commands/optimize.md
```

3. Use your custom command in Claude Code:
```bash
claude > /project:optimize
```

**Tips:**
- Command names are derived from the filename (e.g., `optimize.md` becomes `/project:optimize`)
- You can organize commands in subdirectories
- Project commands are available to everyone who clones the repository
- The Markdown file content becomes the prompt sent to Claude when the command is invoked

### Add command arguments with $ARGUMENTS

**When to use:** You want to create flexible slash commands that can accept additional input from users.

1. Create a command file with the $ARGUMENTS placeholder:
```bash
echo "Find and fix issue #$ARGUMENTS. Follow these steps: 1. Understand the issue described in the ticket 2. Locate the relevant code in our codebase 3. Implement a solution that addresses the root cause 4. Add appropriate tests 5. Prepare a concise PR description" > .claude/commands/fix-issue.md
```

2. Use the command with an issue number:
```bash
claude > /project:fix-issue 123
```

## Run parallel Claude Code sessions with Git worktrees

### Use worktrees for isolated coding environments

**When to use:** You need to work on multiple tasks simultaneously with complete code isolation between Claude Code instances.

1. Create a new worktree:
```bash
# Create a new worktree with a new branch
git worktree add ../project-feature-a feature-a

# Or create a worktree with an existing branch
git worktree add ../project-bugfix bugfix-123
```

2. Run Claude Code in each worktree:
```bash
# Navigate to your worktree
cd ../project-feature-a
# Run Claude Code in this isolated environment
claude
```

3. In another terminal:
```bash
cd ../project-bugfix
claude
```

4. Manage your worktrees:
```bash
# List all worktrees
git worktree list
# Remove a worktree when done
git worktree remove ../project-feature-a
```

**Tips:**
- Each worktree has its own independent file state, making it perfect for parallel Claude Code sessions
- Changes made in one worktree won't affect others, preventing Claude instances from interfering with each other
- All worktrees share the same Git history and remote connections
- For long-running tasks, you can have Claude working in one worktree while you continue development in another