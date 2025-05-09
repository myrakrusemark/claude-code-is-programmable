# Claude Code: Advanced Techniques for AI/Agentic Coding

## Quick Techniques Guide

1. **Context Engineering**: Instead of just prompt engineering, focus on the entire context for the AI model:
   - Create and refine CLAUDE.md files to provide consistent guidelines
   - Use thinking commands (`think`, `think hard`, `think harder`, `ultrathink`) to trigger deeper analysis
   - Mention specific files and use tab-completion for accurate file references
   - Use images and URLs alongside your prompts for richer context

2. **Workflow Patterns**:
   - **Explore → Plan → Code → Commit**: Make Claude read and understand before implementing
   - **Tests → Commit → Code → Iterate → Commit**: Test-driven development with AI
   - **Code → Screenshot → Iterate**: Visual feedback loops for UI development
   - **Safe YOLO Mode**: For trusted operations in safe environments

3. **Multi-Claude Approaches**:
   - Writer/Reviewer Pattern: One Claude writes, another reviews
   - Parallel Processing: Multiple Claude instances working on different parts of a project
   - Git Worktrees: Different instances on different branches

4. **Headless Automation**:
   - Issue triage
   - Custom linting
   - Large-scale migrations via fan-out pattern
   - Data pipeline integration

5. **Tool Extension**:
   - Customize allowlists for operations like editing and git commands
   - Install and document custom CLI tools
   - Connect MCP servers for specialized capabilities
   - Create custom slash commands for repeated workflows

6. **Optimization Techniques**:
   - Be specific in instructions
   - Course-correct early with interrupts (Escape key)
   - Use `/clear` to keep context focused
   - Create checklists for complex multi-stage tasks

---

## 1. Context Engineering Approach

### Creating Effective CLAUDE.md Files

CLAUDE.md files are automatically included in your context and can dramatically improve Claude's effectiveness. These files can be placed in:

- Repository root (most common)
- Parent directories (useful for monorepos)
- Child directories (loaded on demand)
- Home folder (~/.claude/CLAUDE.md) for session-wide settings

**Example CLAUDE.md Content:**
```markdown
# Bash commands
- npm run build: Build the project
- npm run typecheck: Run the typechecker

# Code style
- Use ES modules (import/export) syntax, not CommonJS (require)
- Destructure imports when possible (eg. import { foo } from 'bar')

# Workflow
- Be sure to typecheck when you're done making a series of code changes
- Prefer running single tests, not the whole test suite, for performance
```

**Tips for Effective CLAUDE.md Files:**
- Keep them concise and human-readable
- Iterate on effectiveness, like any prompt
- Use emphasis words like "IMPORTANT" or "YOU MUST" for critical instructions
- Add content while working using the `#` key
- Include CLAUDE.md in commits to benefit your team

### Customizing Tool Allowlists

By default, Claude Code requests permission for system-modifying actions. You can customize what's allowed:

- Select "Always allow" during a session
- Use `/allowed-tools` command
- Edit your `.claude/settings.json` or `~/.claude.json`
- Use the `--allowedTools` CLI flag

### Using Extended Thinking

Claude Code has special thinking mode triggers that allocate progressively more computation time:
- `think` < `think hard` < `think harder` < `ultrathink`

Explicitly mentioning these in your prompts gives Claude more time to consider alternatives and develop plans.

## 2. Workflow Patterns

### Explore, Plan, Code, Commit

This versatile workflow works for many problems:

1. Ask Claude to read relevant files, images, or URLs without writing code yet
2. Request a plan, using thinking mode triggers as needed
3. Have Claude implement the solution in code
4. Ask Claude to commit and create a pull request

**Key insight**: Steps 1-2 are crucial for complex tasks, as they prevent Claude from jumping straight to coding without proper understanding.

### Test-Driven Development (TDD)

1. Ask Claude to write tests based on expected input/output pairs
2. Have Claude run the tests to confirm they fail
3. Ask Claude to commit the tests
4. Request code implementation that passes the tests
5. Commit the passing implementation

This approach is particularly effective because Claude performs best when it has a clear target to iterate against.

### Visual Development Loop

1. Give Claude a way to take screenshots (MCP servers, manual screenshots)
2. Provide a visual mock or design reference
3. Have Claude implement the design, take screenshots, and iterate
4. Commit when satisfied

**Pro tip**: Claude's outputs typically improve significantly with 2-3 iterations of visual feedback.

### Safe YOLO Mode

For trusted operations in controlled environments:
- Use `claude --dangerously-skip-permissions` to bypass permission checks
- Best used for routine tasks like fixing lint errors or generating boilerplate
- For safety, run in a container without internet access

## 3. Multi-Claude Workflows

### Writer/Reviewer Pattern

Run multiple Claude instances with different roles:

1. Have one Claude write code
2. Run `/clear` or start a second Claude in another terminal
3. Have the second Claude review the first Claude's work
4. Start another Claude to integrate feedback and improve the code

This pattern mimics human code review and often produces better results than a single Claude trying to do everything.

### Parallel Processing

To work on multiple independent tasks simultaneously:

1. Create 3-4 git checkouts in separate folders
2. Open each folder in separate terminal tabs
3. Start Claude in each folder with different tasks
4. Cycle through to check progress and approve/deny permission requests

### Git Worktrees

A lighter-weight alternative to multiple checkouts:

1. Create worktrees: `git worktree add ../project-feature-a feature-a`
2. Launch Claude in each worktree: `cd ../project-feature-a && claude`
3. Create additional worktrees as needed
4. Clean up when finished: `git worktree remove ../project-feature-a`

## 4. Headless Automation

Claude Code's headless mode (`claude -p`) enables programmatic integration:

### Fan-out Pattern for Large-Scale Tasks

1. Have Claude write a script to generate a task list
2. Loop through tasks, calling Claude programmatically for each
3. Process results and collect metrics

Example command:
```bash
claude -p "migrate foo.py from React to Vue. When done, return OK or FAIL" --allowedTools Edit Bash
```

### Pipeline Integration

Integrate Claude into data processing pipelines:
```bash
claude -p "<your prompt>" --json | your_command
```

### Automated Issue Management

Use Claude to triage GitHub issues, assign labels, and suggest fixes automatically when issues are created.

### Custom Linting

Claude can provide subjective code reviews beyond traditional linters, identifying:
- Typos
- Stale comments
- Misleading function or variable names
- Inconsistent code styles

## 5. Tool Extension Strategies

### Using Custom Bash Tools

Claude inherits your bash environment and can use your custom tools:
- Tell Claude the tool name with usage examples
- Have Claude run `--help` to see documentation
- Document frequently used tools in CLAUDE.md

### MCP Integration

Claude Code functions as both an MCP server and client:
- Add MCP servers to project config
- Configure in global config
- Include in a checked-in `.mcp.json` file

For debugging, launch Claude with the `--mcp-debug` flag.

### Custom Slash Commands

For repeated workflows, store prompt templates in the `.claude/commands` folder:
- These become available through the slash commands menu
- Can include the `$ARGUMENTS` keyword for parameterization
- Can be checked into git for team sharing

**Example Slash Command Template:**
```markdown
Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:
1. Use `gh issue view` to get the issue details
2. Understand the problem described in the issue
3. Search the codebase for relevant files
4. Implement the necessary changes to fix the issue
5. Write and run tests to verify the fix
6. Ensure code passes linting and type checking
7. Create a descriptive commit message
8. Push and create a PR
```

## 6. Optimization Strategies

### Being Specific in Instructions

| Poor                                             | Good                                                                                               |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| add tests for foo.py                             | write a new test case for foo.py, covering the edge case where the user is logged out. avoid mocks |
| why does ExecutionFactory have such a weird api? | look through ExecutionFactory's git history and summarize how its api came to be                   |

### Course Correction Tools

- Ask Claude to make a plan before coding
- Press Escape to interrupt during any phase
- Double-tap Escape to jump back in history
- Ask Claude to undo changes

### Using /clear for Context Management

During long sessions, use the `/clear` command frequently between tasks to reset the context window and maintain focus.

### Checklists for Complex Workflows

For large tasks:
1. Tell Claude to create a Markdown checklist of subtasks
2. Instruct Claude to address each issue one by one
3. Have Claude check off items as they're completed

### Working with Visual Data

Claude excels with images and diagrams through:
- Paste screenshots (macOS: `cmd+ctrl+shift+4` then `ctrl+v`)
- Drag and drop images directly into the prompt
- Provide file paths for images

## 7. Specialized Workflows

### Git and GitHub Operations

Claude can effectively handle:
- Searching git history
- Writing commit messages
- Handling complex git operations
- Creating pull requests
- Implementing code review fixes
- Fixing failing builds
- Categorizing and triaging issues

### Working with Jupyter Notebooks

- Have Claude Code and a .ipynb file open side-by-side
- Claude can interpret outputs, including images
- Ask Claude to clean up or make aesthetic improvements
- Tell Claude to make notebooks "aesthetically pleasing"

### Codebase Q&A

Claude excels at answering questions about codebases:
- How does [feature] work?
- How do I make a new [component]?
- What does this code do?
- What edge cases are handled?
- Why is the code structured this way?

This approach significantly improves onboarding time and reduces load on other engineers.

---

## Typescript example

```
async function runClaude(prompt: string, dir: string, allowedTools: string, outputFormat?: string): Promise<string> {
  console.log(`🔹 Running Claude in ${dir}...`);
  
  const outputFormatFlag = outputFormat ? `--output-format ${outputFormat}` : '';
  const command = `cd "${dir}" && claude -p "${prompt}" --allowedTools "${allowedTools}" ${outputFormatFlag}`;
  
  try {
    const { stdout, stderr } = await execAsync(command);
    const logFile = join(dir, 'claude_output.log');
    writeFileSync(logFile, stdout);
    return stdout;
  } catch (error) {
    console.error(`Error running Claude: ${error}`);
    return '';
  }
}
```