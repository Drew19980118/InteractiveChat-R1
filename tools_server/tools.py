"""
IGPO Tool Server - Tool Definitions (Web Search Only)
"""

import copy
from typing import Dict, Any

# Web Search Tool Definition
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for information using Google or Bing.",
    "inputs": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of search queries (1-3 queries recommended)"
        }
    },
    "example": {"query": ["What is the capital of France?", "Paris population 2024"]}
}


def get_tools(config: Dict[str, Any] = None) -> Dict[str, Dict]:
    """Get tools with a prompt schema matching the configured query cap."""
    web_search_tool = copy.deepcopy(WEB_SEARCH_TOOL)
    try:
        max_search_queries = int((config or {}).get("max_search_queries", 3))
    except (TypeError, ValueError):
        max_search_queries = 3

    if max_search_queries == 1:
        web_search_tool["inputs"]["query"]["description"] = (
            "A JSON list containing exactly one non-empty search query."
        )
        web_search_tool["example"] = {"query": ["What is the capital of France?"]}

    return {"web_search": web_search_tool}


def get_tool_names(config: Dict[str, Any] = None) -> list:
    """Get list of available tool names."""
    return ["web_search"]
