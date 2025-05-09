#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "anthropic",
#   "rich",
#   "python-dotenv",
# ]
# ///

import argparse
import os
import json
import sys
from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from dotenv import load_dotenv

# Initialize console for rich output
console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search the web using Anthropic's Claude."
    )
    parser.add_argument("query", help="The search query")
    parser.add_argument(
        "--max-uses", type=int, default=3, help="Maximum number of web searches"
    )
    parser.add_argument(
        "--model", default="claude-3-7-sonnet-20250219", help="Claude model to use"
    )
    parser.add_argument("--domains", help="Comma-separated list of allowed domains")
    parser.add_argument("--blocked", help="Comma-separated list of blocked domains")
    parser.add_argument(
        "--location", help="Location format: 'US,California,San Francisco'"
    )
    parser.add_argument(
        "--timezone", help="IANA timezone ID (e.g., 'America/Los_Angeles')"
    )
    return parser.parse_args()


def parse_location(location_string, timezone=None):
    """Parse location string into the required format."""
    if not location_string:
        return None

    parts = location_string.split(",")
    if len(parts) < 3:
        console.print(
            "[yellow]Warning:[/] Location must have country, region, and city"
        )
        return None

    location = {
        "type": "approximate",
        "country": parts[0].strip(),
        "region": parts[1].strip(),
        "city": parts[2].strip(),
    }

    if timezone:
        location["timezone"] = timezone

    return location


def get_search_results(args):
    """Query Anthropic API with web search enabled."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Parse domain lists
    allowed_domains = args.domains.split(",") if args.domains else None
    blocked_domains = args.blocked.split(",") if args.blocked else None

    # Build the web search tool
    web_search_tool = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": args.max_uses,
    }

    # Add domain filtering if specified (can't use both)
    if allowed_domains:
        web_search_tool["allowed_domains"] = allowed_domains
    elif blocked_domains:
        web_search_tool["blocked_domains"] = blocked_domains

    # Add location if specified
    location = parse_location(args.location, args.timezone)
    if location:
        web_search_tool["user_location"] = location

    try:
        # Make API request
        response = client.messages.create(
            model=args.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": args.query}],
            tools=[web_search_tool],
        )
        return response
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return None


def extract_sources(response):
    """Extract sources from the response."""
    sources = []
    url_set = set()  # To track unique URLs

    for block in response.content:
        if (
            block.type == "text"
            and hasattr(block, "citations")
            and block.citations is not None
        ):
            for citation in block.citations:
                if (
                    citation.type == "web_search_result_location"
                    and citation.url not in url_set
                ):
                    url_set.add(citation.url)
                    sources.append(
                        {
                            "url": citation.url,
                            "title": citation.title,
                            "text": citation.cited_text,
                        }
                    )

    return sources


def format_search_usage(response):
    """Format the search usage information."""
    usage_text = ""

    if hasattr(response, "usage") and hasattr(response.usage, "server_tool_use"):
        server_tool_use = response.usage.server_tool_use
        if hasattr(server_tool_use, "web_search_requests"):
            usage_text = f"Searches performed: {server_tool_use.web_search_requests}"

    return usage_text


def display_results(query, response, sources):
    """Display search results with citations."""
    # Create source mapping for citation numbering
    source_map = {source["url"]: i + 1 for i, source in enumerate(sources)}

    # Print query
    console.print(Panel(f"[bold]Search:[/] {query}", border_style="blue"))

    # Extract and format the response text
    response_text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text

            # Add citation numbers
            if hasattr(block, "citations") and block.citations is not None:
                for citation in block.citations:
                    if citation.type == "web_search_result_location":
                        url = citation.url
                        citation_num = source_map[url]
                        # Add citation number after cited text
                        text = text.replace(
                            citation.cited_text,
                            f"{citation.cited_text}[{citation_num}]",
                        )

            response_text += text + " "

    # Print formatted response
    console.print(Panel(response_text.strip(), title="Response", border_style="green"))

    # Print sources
    if sources:
        sources_text = "\n".join(
            f"[{i+1}] {source['url']} - {source['title']}"
            for i, source in enumerate(sources)
        )
        console.print(Panel(sources_text, title="Sources", border_style="yellow"))

    # Print usage information
    usage_info = format_search_usage(response)
    if usage_info:
        console.print(Panel(usage_info, title="Usage", border_style="cyan"))


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/] ANTHROPIC_API_KEY environment variable not found"
        )
        console.print("Please set your API key using one of the following methods:")
        console.print("1. Create a .env file with ANTHROPIC_API_KEY=your-api-key")
        console.print("2. Export ANTHROPIC_API_KEY=your-api-key in your shell")
        sys.exit(1)

    # Parse arguments
    args = parse_args()

    # Display search query
    console.print(f"[bold]Searching for:[/] {args.query}")

    # Get search results
    response = get_search_results(args)
    if not response:
        sys.exit(1)

    # Extract sources
    sources = extract_sources(response)

    # Display results
    display_results(args.query, response, sources)


if __name__ == "__main__":
    main()
