"""
IGPO Tool Server - Web Search API

Supports:
- Serper API (Google search)
- Azure Bing Search API
- A local retriever compatible with the IGPO QReCC index service

Reference: DeepResearcher/scrl/handler/web_search_agent/search/search_api.py
"""

import json
import time
import http.client
import requests
from typing import List, Dict, Any, Optional


def web_search(query: str, config: Dict[str, Any]) -> List[Dict]:
    """
    Unified web search function.
    
    Args:
        query: Search query string
        config: Configuration dictionary with search engine settings
        
    Returns:
        List of search results with 'title', 'link', 'snippet' keys
    """
    if not query:
        raise ValueError("Search query cannot be empty")
    
    # Mock mode for testing without network
    if config.get('mock_mode', False):
        return mock_search_results(query)
    
    search_engine = config.get('search_engine', 'google')
    
    if search_engine == 'google':
        return serper_google_search(
            query=query,
            serper_api_key=config.get('serper_api_key', ''),
            top_k=config.get('search_top_k', 10),
            region=config.get('search_region', 'us'),
            lang=config.get('search_lang', 'en')
        )
    elif search_engine == 'bing':
        return azure_bing_search(
            query=query,
            subscription_key=config.get('azure_bing_search_subscription_key', ''),
            mkt=config.get('azure_bing_search_mkt', 'en-US'),
            top_k=config.get('search_top_k', 10)
        )
    elif search_engine == 'local_retriever':
        return local_retriever_search(
            query=query,
            url=config.get('local_retriever_url', 'http://127.0.0.1:8002/retrieve'),
            top_k=config.get('search_top_k', 10),
            timeout=config.get('local_retriever_timeout', 30),
            max_retries=config.get('local_retriever_max_retries', 3),
            retry_delay=config.get('local_retriever_retry_delay', 2),
        )
    else:
        raise ValueError(f"Unknown search engine: {search_engine}")


def serper_google_search(
    query: str,
    serper_api_key: str,
    top_k: int = 10,
    region: str = "us",
    lang: str = "en",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict]:
    """
    Search using Serper API (Google search).
    
    Get your API key from: https://serper.dev/
    
    Args:
        query: Search query
        serper_api_key: Serper API key
        top_k: Number of results to return
        region: Search region (e.g., 'us', 'cn')
        lang: Search language (e.g., 'en', 'zh')
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        
    Returns:
        List of search results
    """
    if not serper_api_key:
        print("[WARNING] Serper API key not configured. Returning empty results.")
        return []
    
    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
            headers = {
                'X-API-KEY': serper_api_key,
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            
            if not data:
                raise Exception("Empty response from Serper API")
            
            if "organic" not in data:
                print(f"[WARNING] No organic results for query: '{query}'")
                return []
            
            results = data["organic"]
            print(f"[Search] Serper search success: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"[Search] Serper API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    print(f"[Search] Serper search failed after {max_retries} attempts")
    return []


def azure_bing_search(
    query: str,
    subscription_key: str,
    mkt: str = "en-US",
    top_k: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict]:
    """
    Search using Azure Bing Search API.
    
    Get your key from: https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/
    
    Args:
        query: Search query
        subscription_key: Azure subscription key
        mkt: Market code (e.g., 'en-US', 'zh-CN')
        top_k: Number of results to return
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        
    Returns:
        List of search results with 'title', 'link', 'snippet' keys
    """
    if not subscription_key:
        print("[WARNING] Azure Bing subscription key not configured. Returning empty results.")
        return []
    
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                "https://api.bing.microsoft.com/v7.0/search",
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            json_response = response.json()
            
            results = []
            if 'webPages' in json_response and 'value' in json_response['webPages']:
                for e in json_response['webPages']['value']:
                    results.append({
                        "title": e.get('name', ''),
                        "link": e.get('url', ''),
                        "snippet": e.get('snippet', '')
                    })
            
            print(f"[Search] Bing search success: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"[Search] Bing API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    print(f"[Search] Bing search failed after {max_retries} attempts")
    return []


def local_retriever_search(
    query: str,
    url: str = "http://127.0.0.1:8002/retrieve",
    top_k: int = 10,
    timeout: float = 30,
    max_retries: int = 3,
    retry_delay: float = 2,
) -> List[Dict]:
    """Search the local retriever and adapt its response to IGPO's tool schema.

    The local service is expected to accept::

        {"queries": ["..."], "topk": 10, "return_scores": true}

    and return ``{"result": [[{"document": {...}, "score": ...}]]}``.
    A single query is intentionally wrapped in ``queries`` because that is the
    API contract of the local service.
    """
    try:
        top_k = max(1, int(top_k))
        timeout = float(timeout)
        max_retries = max(1, int(max_retries))
        retry_delay = float(retry_delay)
    except (TypeError, ValueError):
        print("[Search] Invalid local retriever configuration. Returning empty results.")
        return []

    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": True,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()

            raw_results = body.get("result") if isinstance(body, dict) else None
            if not isinstance(raw_results, list) or not raw_results:
                raise ValueError("response must contain a non-empty list in 'result'")

            query_results = raw_results[0]
            if not isinstance(query_results, list):
                raise ValueError("response['result'][0] must be a list")

            results = []
            for rank, item in enumerate(query_results, start=1):
                if not isinstance(item, dict) or not isinstance(item.get("document"), dict):
                    print(f"[Search] Ignoring malformed local retriever result at rank {rank}")
                    continue

                document = item["document"]
                passage_id = document.get("passage_id", "")
                title = document.get("title", "")
                passage_text = document.get("passage_text", "")

                results.append({
                    "title": str(title) if title is not None else "",
                    "link": f"local://passage/{passage_id}" if passage_id not in (None, "") else "",
                    "snippet": str(passage_text) if passage_text is not None else "",
                    "passage_id": passage_id,
                    "score": item.get("score"),
                })


            return results
        except (requests.RequestException, ValueError) as error:
            print(
                f"[Search] Local retriever error "
                f"(attempt {attempt + 1}/{max_retries}): {error}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    print(f"[Search] Local retriever failed after {max_retries} attempts")
    return []


def mock_search_results(query: str) -> List[Dict]:
    """
    Generate mock search results for testing without network.
    
    Args:
        query: Search query string
        
    Returns:
        List of mock search results
    """
    print(f"[Search] Mock mode: generating fake results for '{query}'")
    return [
        {
            "title": f"Mock Result 1 for: {query}",
            "link": f"https://example.com/result1?q={query.replace(' ', '+')}",
            "snippet": f"This is a mock search result for the query '{query}'. It contains relevant information about the topic."
        },
        {
            "title": f"Mock Result 2 for: {query}",
            "link": f"https://example.org/result2?q={query.replace(' ', '+')}",
            "snippet": f"Another mock result providing details about '{query}'. This simulates real search engine output."
        },
        {
            "title": f"Mock Result 3 for: {query}",
            "link": f"https://test.com/result3?q={query.replace(' ', '+')}",
            "snippet": f"Third mock result for '{query}'. Use mock_mode=True in config to enable this feature."
        }
    ]


if __name__ == "__main__":
    # Test search
    print("Testing Serper Google search...")
    results = serper_google_search("test query", "your_api_key_here", top_k=3)
    print(f"Results: {results}")
