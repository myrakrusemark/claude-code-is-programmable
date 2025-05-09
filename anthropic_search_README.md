# Anthropic Web Search Tool

A command-line utility for searching the web using Anthropic's Claude AI with their web search tool capability.

## Prerequisites

- Python 3.8+
- UV package manager (`pip install uv`)
- Anthropic API key

## Setup

1. Create a `.env` file in the same directory as the script with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your-api-key
   ```
   
   Or export it directly in your terminal:
   ```
   export ANTHROPIC_API_KEY=your-api-key
   ```

2. Make the script executable:
   ```
   chmod +x anthropic_search.py
   ```

## Usage

Basic search:
```
./anthropic_search.py "your search query"
```

With domain filtering (only include results from these domains):
```
./anthropic_search.py "javascript best practices" --domains "developer.mozilla.org,javascript.info"
```

Block specific domains:
```
./anthropic_search.py "climate change" --blocked "unreliablesource.com,fakenews.org"
```

With location context:
```
./anthropic_search.py "local restaurants" --location "US,California,San Francisco" --timezone "America/Los_Angeles"
```

Increase maximum searches:
```
./anthropic_search.py "complex research topic" --max-uses 5
```

Use a different Claude model:
```
./anthropic_search.py "your query" --model "claude-3-5-sonnet-latest"
```

## Output

The script produces:
1. The search query used
2. Claude's response with inline citations marked as [1], [2], etc.
3. A list of sources at the end, numbered to match the citations
4. Usage information showing how many web searches were performed

## Notes

- Web search is available on Claude 3.7 Sonnet, Claude 3.5 Sonnet, and Claude 3.5 Haiku
- Each search counts as one use, regardless of the number of results returned
- Searches cost $10 per 1,000 searches, plus standard token costs for search-generated content
- Domain filtering doesn't need https:// prefixes and automatically includes subdomains