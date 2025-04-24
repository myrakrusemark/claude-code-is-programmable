#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "openai",
#   "openai-agents",
#   "python-dotenv"
# ]
# ///

import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner
from agents.mcp.server import MCPServerStdio

# Load environment variables
load_dotenv()
NOTION_API_SECRET = os.getenv("NOTION_INTERNAL_INTEGRATION_SECRET")

if not NOTION_API_SECRET:
    print("ERROR: NOTION_INTERNAL_INTEGRATION_SECRET not found in environment variables.")
    exit(1)

async def main():
    # Set up Notion MCP server
    print("Setting up Notion MCP server...")
    
    # Configure headers with the Notion API token and version
    headers_json = f'{{"Authorization": "Bearer {NOTION_API_SECRET}", "Notion-Version": "2022-06-28"}}'
    
    # Create the Notion MCP server
    async with MCPServerStdio(
        name="Notion API Server",
        params={
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "env": {
                "OPENAPI_MCP_HEADERS": headers_json
            }
        }
    ) as notion_server:
        print("Notion MCP server started")
        
        # Create a simple agent with access to the Notion MCP server
        agent = Agent(
            name="Notion Test Agent",
            instructions="""
            You are a test agent with access to the Notion API.
            Use the tools available to you to search for a page and report its ID.
            """,
            mcp_servers=[notion_server]
        )
        
        # List available tools to verify connection
        tools = await notion_server.list_tools()
        print(f"Found {len(tools)} tools available from Notion MCP server")
        
        # Run a simple test query
        print("\nRunning test query...")
        result = await Runner.run(
            agent,
            "Search for a page named 'Agentic Task List' and tell me its ID."
        )
        
        print("\n=== Test Result ===")
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())