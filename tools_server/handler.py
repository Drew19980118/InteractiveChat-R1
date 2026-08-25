"""
IGPO Tool Server - Handler (Web Search Only)

Lightweight handler for web search tool calls.
Supports Serper API (Google) and Azure Bing Search.
"""

import os
import json
import time
import threading
import concurrent.futures
from typing import List, Dict, Any

from tools_server.search.search_api import web_search


class Handler:
    """Web search handler with local caching."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = config.get('cache_dir', './cache/tool_cache')
        self.cache_ttl = config.get('cache_ttl_days', 7) * 24 * 60 * 60
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(self.cache_dir, 'search_cache.json')
        self.search_cache = self._load_cache()
        self.cache_lock = threading.Lock()
    
    def _load_cache(self) -> Dict:
        """Load search cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        """Save search cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Handler] Cache save error: {e}")
    
    def _is_cache_valid(self, entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return entry.get('timestamp', 0) and (time.time() - entry['timestamp']) < self.cache_ttl

    def _cache_key(self, query: str) -> str:
        """Keep cached results from different backends and result depths separate."""
        top_k = int(self.config.get('search_top_k', 10))
        if self.config.get('mock_mode', False):
            return f"mock::topk={top_k}::{query}"

        engine = self.config.get('search_engine', 'google')
        if engine == 'local_retriever':
            endpoint = self.config.get('local_retriever_url', 'http://127.0.0.1:8002/retrieve')
            return f"{engine}::{endpoint}::topk={top_k}::{query}"
        return f"{engine}::topk={top_k}::{query}"

    def _max_search_queries(self) -> int:
        """Return the per-tool-call query cap configured for this experiment."""
        return max(1, int(self.config.get('max_search_queries', 3)))

    def _search_top_k(self) -> int:
        """Return the number of passages allowed in the tool observation."""
        return max(1, int(self.config.get('search_top_k', 10)))
    
    def handle_all(self, task_list: List[Dict]) -> List[Dict]:
        """Process all web search tasks."""
        if not task_list:
            return task_list
        
        print(f"[Handler] Processing {len(task_list)} tasks...")
        start_time = time.time()
        
        # Pre-fetch all search queries
        self._prefetch_searches(task_list)
        
        # Process each task
        for task in task_list:
            tool_call = task.get('tool_call', {})
            # Ensure tool_call is a dict
            if not isinstance(tool_call, dict):
                task['content'] = f"Invalid tool_call format: expected dict, got {type(tool_call).__name__}"
                continue
            
            tool_name = tool_call.get('name', '')
            if tool_name == 'web_search':
                arguments = tool_call.get('arguments', {})
                if not isinstance(arguments, dict):
                    arguments = {}
                task['content'] = self._handle_web_search(arguments)
            else:
                task['content'] = f"Unknown tool: {tool_name}"
        
        print(f"[Handler] Completed in {time.time() - start_time:.2f}s")
        return task_list
    
    def _prefetch_searches(self, task_list: List[Dict]):
        """Pre-fetch all search queries in parallel."""
        queries_to_fetch = set()
        
        for task in task_list:
            tool_call = task.get('tool_call', {})
            # Ensure tool_call is a dict
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get('name') != 'web_search':
                continue
            
            arguments = tool_call.get('arguments', {})
            if not isinstance(arguments, dict):
                continue
            
            query_list = arguments.get('query', [])
            if not isinstance(query_list, list):
                query_list = [query_list] if query_list else []
            
            for query in query_list[:self._max_search_queries()]:
                if isinstance(query, str):
                    cache_key = self._cache_key(query)
                    with self.cache_lock:
                        if cache_key not in self.search_cache or not self._is_cache_valid(self.search_cache[cache_key]):
                            queries_to_fetch.add(query)
        
        if not queries_to_fetch:
            return
        
        print(f"[Handler] Fetching {len(queries_to_fetch)} search queries...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(web_search, q, self.config): q for q in queries_to_fetch}
            for future in concurrent.futures.as_completed(futures):
                query = futures[future]
                try:
                    results = future.result(timeout=30)
                    with self.cache_lock:
                        self.search_cache[self._cache_key(query)] = {'timestamp': time.time(), 'results': results}
                except Exception as e:
                    print(f"[Handler] Search error for '{query}': {e}")
        
        self._save_cache()
    
    def _handle_web_search(self, arguments: Dict) -> List[Dict]:
        """Handle web_search tool call."""
        query_list = arguments.get('query', [])
        
        # Handle both single string and list of strings
        if isinstance(query_list, str):
            query_list = [query_list]
        elif not isinstance(query_list, list):
            return []
        
        results = []
        for query in query_list[:self._max_search_queries()]:
            if not isinstance(query, str):
                continue
            
            # Get from cache or fetch
            cache_key = self._cache_key(query)
            with self.cache_lock:
                entry = self.search_cache.get(cache_key, {})
                if self._is_cache_valid(entry):
                    search_results = entry.get('results', [])
                else:
                    search_results = web_search(query, self.config)
                    self.search_cache[cache_key] = {'timestamp': time.time(), 'results': search_results}
            
            # Format results
            results.append({
                "search_query": query,
                "web_page_info_list": [
                    {
                        "title": r.get('title', ''),
                        "url": r.get('link', r.get('url', '')),
                        "quick_summary": r.get('snippet', r.get('description', '')),
                        **({"passage_id": r["passage_id"]} if "passage_id" in r else {}),
                        **({"score": r["score"]} if r.get("score") is not None else {}),
                    }
                    for r in search_results[:self._search_top_k()]
                ]
            })
        
        return results
