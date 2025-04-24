#!/usr/bin/env -S uv run
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "openai",
#   "openai-agents",
#   "pydantic",
#   "rich",
#   "python-dotenv"
# ]
# ///

import os
import sys
import asyncio
import subprocess
from typing import Any, List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from dotenv import load_dotenv
from pydantic import BaseModel

from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings
from agents.mcp.server import MCPServer
from agents.mcp.server import MCPServerStdio

# Initialize rich console
console = Console()

# Constants
MODEL = "o4-mini"  # OpenAI model to use for all agents

# Load environment variables
load_dotenv()
NOTION_API_SECRET = os.getenv("NOTION_INTERNAL_INTEGRATION_SECRET")
if not NOTION_API_SECRET:
    console.print(
        "[bold red]ERROR: NOTION_INTERNAL_INTEGRATION_SECRET not found in environment variables.[/bold red]"
    )
    sys.exit(1)


# Global variable to store the singleton Notion MCP server instance
_notion_mcp_server = None


# Define our MCP server for Notion API with singleton pattern
async def get_notion_mcp_server() -> MCPServer:
    global _notion_mcp_server

    # If server is already created, return the existing instance
    if _notion_mcp_server is not None:
        return _notion_mcp_server

    # Otherwise, create a new server instance
    console.print(
        "[bold blue]📡 Setting up Notion MCP server (first time)...[/bold blue]"
    )

    # Configure headers with the Notion API token and version
    headers_json = f'{{"Authorization": "Bearer {NOTION_API_SECRET}", "Notion-Version": "2022-06-28"}}'

    # Create and store the Notion MCP server
    _notion_mcp_server = MCPServerStdio(
        name="Notion API Server",
        params={
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "env": {"OPENAPI_MCP_HEADERS": headers_json},
        },
    )

    await _notion_mcp_server.connect()

    return _notion_mcp_server


# Define the data classes for our agents
class TodoItem(BaseModel):
    id: str
    content: str
    is_completed: bool


class GetNotionPageContent(BaseModel):
    """Content of a Notion page, including raw content and todo items"""

    raw_content: str
    todo_items: List[TodoItem]

    def __str__(self) -> str:
        """String representation of the content"""
        todos_str = "\n".join(
            [
                f"- {'[x]' if item.is_completed else '[ ]'} {item.content} (ID: {item.id})"
                for item in self.todo_items
            ]
        )
        return f"Page Content:\n{self.raw_content}\n\nTodo Items ({len(self.todo_items)}):\n{todos_str}"


class TodoItems(BaseModel):
    """Collection of TodoItem objects"""

    items: List[TodoItem]


class TodoUpdateResult(BaseModel):
    """Result of updating a todo item"""

    success: bool
    message: str
    todo_id: str


# Helper function to run Claude Code
def claude_code(prompt: str) -> str:
    """
    Run Claude Code with a prompt and return the output.

    Args:
        prompt: The AI prompt describing what code to generate

    Returns:
        The output from Claude Code
    """
    try:
        cmd = [
            "claude",
            "-p",
            prompt,
            "--allowedTools",
            "Edit",
            "Bash",
            "Write",
        ]

        # Run the command and capture output
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Return the output from the process
        return process.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing Claude Code (return code {e.returncode}): {e.stderr}"
    except Exception as e:
        return f"Unexpected error running Claude Code: {str(e)}"


# Tool implementations
@function_tool
async def find_notion_page(page_name: str) -> str:
    """
    Find a Notion page ID based on its name.

    Args:
        page_name: The name of the Notion page to find

    Returns:
        The page ID if found, or an error message
    """
    console.print(
        f"[bold cyan]BEGIN --- find_notion_page(page_name={page_name})[/bold cyan]"
    )

    try:
        # Create a sub-agent with access to the Notion MCP server
        notion_search_agent = Agent(
            name="Notion Page Finder",
            model=MODEL,
            instructions="""
            You are a specialized agent for finding Notion pages.
            Your task is to search for a specific page by name and return its ID.
            Use the Notion API tools provided to you to search for the page.
            If multiple pages match, return the most relevant one.
            If no pages match, return a clear error message.
            
            IMPORTANT: Return ONLY the page ID as a string without any additional text or formatting.
            For example, if you find a page with ID "1e0fc382-ac73-806e-a28d-cc99f7d75096", just return that ID.
            """,
            mcp_servers=[await get_notion_mcp_server()],
        )

        # Run the agent to find the page - using a simple print instead of rich status
        print(f"🔍 Searching for Notion page: {page_name}...")
        result = await Runner.run(
            notion_search_agent, f"Find the Notion page with the name: {page_name}"
        )
        print("✓ Search complete")

        # Extract the page ID from the result
        page_id = result.final_output.strip()

        # Log the result
        result_str = f"Found page with ID: {page_id}"
        console.print(
            f"[bold cyan]END --- find_notion_page(page_name={page_name}) -> {result_str}[/bold cyan]"
        )

        return page_id

    except Exception as e:
        # Capture and format the error
        error_message = f"ERROR finding Notion page: {str(e)}"
        console.print(f"[bold red]{error_message}[/bold red]")
        console.print(
            f"[bold cyan]END --- find_notion_page(page_name={page_name}) -> [ERROR][/bold cyan]"
        )

        # Return the error as a string with a special prefix to identify it as an error
        return f"ERROR: {error_message}"


@function_tool
async def get_notion_page_content(page_id: str) -> str:
    """
    Get the content of a Notion page, specifically focusing on todo items.

    Args:
        page_id: The ID of the Notion page

    Returns:
        A string representation of the page content and todo items
    """
    console.print(
        f"[bold cyan]BEGIN --- get_notion_page_content(page_id={page_id})[/bold cyan]"
    )

    # Create a sub-agent with access to the Notion MCP server
    notion_content_agent = Agent(
        name="Notion Content Retriever",
        model=MODEL,
        instructions="""
        You are a specialized agent for retrieving Notion page content.
        Your task is to get the content of a specific page by ID and:
        1. Extract the page's raw content as text
        2. Find and extract all todo items on the page
        
        For each todo item, extract:
        - Its ID 
        - Content text
        - Completion status (true if completed, false if not)
        
        Use the Notion API to retrieve the page blocks and look for to_do blocks.
        """,
        output_type=GetNotionPageContent,
        mcp_servers=[await get_notion_mcp_server()],
    )

    # Run the agent to get the page content
    print(f"📄 Retrieving Notion page content for ID: {page_id}...")
    result = await Runner.run(
        notion_content_agent,
        f"Get the content of the Notion page with ID: {page_id}. Return both the raw page content and all todo items found.",
    )
    print("✓ Content retrieval complete")

    # Get the structured result
    try:
        # The agent returns a structured GetNotionPageContent object
        page_content = result.final_output_as(GetNotionPageContent)
        # Convert to string for returning
        content_str = str(page_content)
        result_str = content_str
    except Exception as e:
        console.print(f"[bold red]Error getting page content: {str(e)}[/bold red]")
        # Fallback to error message
        content_str = f"Error retrieving page content: {str(e)}"
        result_str = "0 todo items found (error occurred): " + content_str

    console.print(
        f"[bold cyan]END --- get_notion_page_content(page_id={page_id}) -> {result_str}[/bold cyan]"
    )
    return content_str


@function_tool
async def complete_todo(todo_id: str) -> str:
    """
    Mark a todo item as complete in Notion.

    Args:
        todo_id: The ID of the todo item to mark as complete

    Returns:
        A confirmation message
    """
    console.print(f"[bold cyan]BEGIN --- complete_todo(todo_id={todo_id})[/bold cyan]")

    # Create a sub-agent with access to the Notion MCP server
    notion_update_agent = Agent(
        name="Notion Todo Completer",
        model=MODEL,
        instructions="""
        You are a specialized agent for updating Notion todo items.
        Your task is to mark a specific todo item as complete.
        
        Use the Notion API tools provided to you to:
        1. Update the block with the given ID
        2. Set the "checked" property of the to_do block to true
        
        If there's an error, include details about what went wrong.
        """,
        output_type=TodoUpdateResult,
        mcp_servers=[await get_notion_mcp_server()],
    )

    # Run the agent to mark the todo as complete
    print(f"✅ Marking todo {todo_id} as complete...")
    result = await Runner.run(
        notion_update_agent,
        f"Mark the todo item with ID {todo_id} as complete. This is a to_do block type in Notion.",
    )
    print("✓ Update operation complete")

    # Get the structured result
    try:
        # Get the structured result from the agent
        update_result = result.final_output_as(TodoUpdateResult)
        if update_result.success:
            result_str = update_result.message
        else:
            result_str = f"Failed to mark todo as complete: {update_result.message}"
    except Exception as e:
        # If type conversion fails, use the raw output
        console.print(
            f"[bold yellow]Warning: Could not convert result to TodoUpdateResult: {str(e)}[/bold yellow]"
        )
        result_str = f"Todo update completed with response: {result.final_output}"

    console.print(
        f"[bold cyan]END --- complete_todo(todo_id={todo_id}) -> {result_str}[/bold cyan]"
    )
    return result_str


@function_tool
async def ai_code_with_claude_code(ai_coding_prompt: str) -> str:
    """
    Generate code using Claude Code based on a prompt.

    Args:
        ai_coding_prompt: The precise AI Coding Prompt describing the exact code to generate and where to save it

    Returns:
        The generated code or an error message
    """
    console.print(
        f"[bold cyan]BEGIN --- ai_code_with_claude_code(ai_coding_prompt={ai_coding_prompt})[/bold cyan]"
    )

    # Run Claude Code CLI with the prompt
    print(f"🤖 Generating code with Claude Code...")

    ai_coding_prompt_with_git_instructions = f"""
    ## Process:
    1. Implement the changes detailed in the instructions below.
    2. Git stage, and commit the changes.
    3. Respond with success summarizing the changes or an error message detailing what went wrong.

    ## Instructions:
    {ai_coding_prompt}
    """

    result_str = claude_code(ai_coding_prompt_with_git_instructions)
    print("✓ Code generation complete")

    # Log just the first few characters of the result to avoid console clutter
    preview = result_str[:100] + "..." if len(result_str) > 100 else result_str
    console.print(
        f"[bold cyan]END --- ai_code_with_claude_code(ai_coding_prompt={ai_coding_prompt}) -> {preview}[/bold cyan]"
    )
    return result_str


@function_tool
async def ai_code_parallel_with_claude_code(ai_coding_prompts: List[str]) -> str:
    """
    Generate code using Claude Code by running multiple prompts in parallel.

    Args:
        ai_coding_prompts: List of coding prompts to execute in parallel.
                           IMPORTANT: Prompts should be INDEPENDENT of each other and not have dependencies
                           between them. If there are dependencies, they should be run sequentially using
                           the ai_code_with_claude_code tool instead.

    Returns:
        Concatenated results from all code generation prompts, or error messages if failures occurred
    """
    console.print(
        f"[bold cyan]BEGIN --- ai_code_parallel_with_claude_code(ai_coding_prompts=List[{len(ai_coding_prompts)} prompts])[/bold cyan]"
    )

    # For parallel execution, handle each prompt
    console.print(
        f"[bold green]🚀 Running {len(ai_coding_prompts)} prompts in parallel[/bold green]"
    )

    async def run_prompt(prompt_index: int, prompt: str) -> tuple[int, str]:
        """Run a single prompt and return its index and result."""
        try:
            prompt_short = prompt[:30] + "..." if len(prompt) > 30 else prompt
            print(
                f"🤖 Starting prompt {prompt_index+1}/{len(ai_coding_prompts)}: {prompt_short}"
            )

            # Add git instructions
            prompt_with_git_instructions = f"""
            ## Process:
            1. Implement the changes detailed in the instructions below.
            2. Git stage, and commit the changes.
            3. Respond with success summarizing the changes or an error message detailing what went wrong.

            ## Instructions:
            {prompt}
            """

            # Build command
            cmd = [
                "claude",
                "-p",
                prompt_with_git_instructions,
                "--allowedTools",
                "Edit",
                "Bash",
                "Write",
            ]

            # Run the command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"✓ Prompt {prompt_index+1} completed successfully")
                return prompt_index, stdout.decode()
            else:
                error_msg = (
                    stderr.decode()
                    if stderr
                    else f"Process exited with return code {process.returncode}"
                )
                print(f"⚠️ Prompt {prompt_index+1} failed: {error_msg[:50]}...")
                return prompt_index, f"Error executing Claude Code: {error_msg}"

        except Exception as e:
            print(f"⚠️ Prompt {prompt_index+1} failed with exception: {str(e)[:50]}...")
            return prompt_index, f"Unexpected error running Claude Code: {str(e)}"

    # Create tasks for all prompts
    tasks = [run_prompt(i, prompt) for i, prompt in enumerate(ai_coding_prompts)]

    # Run all tasks in parallel and wait for them to complete
    results = await asyncio.gather(*tasks)

    # Sort results by original index to maintain order
    sorted_results = sorted(results, key=lambda x: x[0])

    # Combine all results with clear separators
    combined_results = []
    for i, (_, result) in enumerate(sorted_results):
        combined_results.append(f"\n\n--- RESULT FROM PROMPT {i+1} ---\n\n{result}")

    result_str = "".join(combined_results)
    print(f"✅ All {len(ai_coding_prompts)} prompts completed")

    console.print(
        f"[bold cyan]END --- ai_code_parallel_with_claude_code({ai_coding_prompts} parallel prompts) -> {result_str}[/bold cyan]"
    )
    return result_str


# Define our main agent
async def create_notion_agent():
    # Primary agent with detailed system prompt and tools
    agent = Agent(
        name="Notion Code Generator",
        model=MODEL,
        instructions="""
        # Notion Code Generator Agent

        ## What you're doing:
        You are a powerful agent that can find a Notion page, get its content, iterate through todos, write code based on each todo ITERATIVELY, and then mark todos as complete.

        ## The process:
        1. You will first find the Notion page based on the provided page name using the find_notion_page tool.
        2. Then you will get the page content using the get_notion_page_content tool.
           - This will return a formatted string with the page content and todo items
           - Each todo item will include an ID that you'll need for later steps
        3. Analyze the todo items and decide how to process them:
           - If the todos are INDEPENDENT of each other (can be implemented in parallel):
             a. Group independent todos and generate code for them in parallel using the 
                ai_code_parallel_with_claude_code tool, passing a list of coding prompts.
             b. After the code is successfully written, mark each todo as complete.
           - If the todos are DEPENDENT on each other (must be done sequentially):
             a. Process them ONE BY ONE using the ai_code_with_claude_code tool.
             b. After each todo's code is successfully written, mark it as complete.
             c. Move on to the next todo until all todos are complete.
           
        ## General guidelines:
        - Follow the process strictly in the order described above.
        - Provide clear updates about your progress through each step.
        - If any step fails, try to recover or provide a detailed explanation of the issue.
        - When parsing the todo list, look for todo items in this format:
          - [x] or [ ] at the start (indicates completion status)
          - Followed by the todo text
          - With (ID: some-id-value) at the end - you'll need this ID for the complete_todo call
        
        ## Parallelization guidelines:
        - CAREFULLY assess dependencies between todos before deciding to run them in parallel.
        - If the todos modify different files or independent parts of the codebase, they can likely run in parallel.
        - If one todo depends on changes made by another todo, they MUST be run sequentially.
        - When in doubt, default to sequential processing.
        - You can combine some sequential and parallel processing if needed.
        
        ## VERY IMPORTANT:
        - Skip todos that are already marked as complete [x].
        - Generate code → mark complete → move to next todo or next batch of todos.
        - When using parallel processing, ensure you verify that the todos are truly independent.
        """,
        tools=[
            find_notion_page,
            get_notion_page_content,
            complete_todo,
            ai_code_with_claude_code,
            ai_code_parallel_with_claude_code,
        ],
    )

    return agent


async def main():
    # Check if page name was provided
    if len(sys.argv) < 2:
        console.print(
            "[bold red]ERROR: Please provide a Notion page name as an argument.[/bold red]"
        )
        console.print(
            "Usage: uv run claude_code_is_programmable_3.py <notion_page_name>"
        )
        sys.exit(1)

    page_name = sys.argv[1]

    # Welcome message
    console.print(
        Panel.fit(
            "[bold blue]🤖 Claude Code Notion Agent[/bold blue]\n\n"
            f"[green]Finding and processing todos from Notion page: [bold]{page_name}[/bold][/green]",
            title="Claude Code is Programmable v3",
            border_style="cyan",
        )
    )

    # Create and run our agent
    agent = await create_notion_agent()

    # Run the agent with the page name as input (using simple prints instead of Progress)
    print("⏳ Running Notion Code Generator Agent...")
    result = await Runner.run(
        agent,
        f"Process all todos on the Notion page named '{page_name}'. Follow the process: find page → get content → iterate through todos (generate code → mark complete) one by one.",
        max_turns=20,
    )
    print("✅ Agent task completed\n")

    # Show final results
    console.print("\n[bold green]✅ Agent execution complete![/bold green]")
    console.print(
        Panel(result.final_output, title="Agent Output", border_style="green")
    )


if __name__ == "__main__":
    asyncio.run(main())
