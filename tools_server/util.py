"""
IGPO Tool Server - Utilities (Web Search Only)
"""

import os
import json
import time
import uuid
import socket
import datetime
import threading
import traceback
from typing import List, Dict, Any

import requests


def _parse_bool_environment(value: str, variable_name: str) -> bool:
    """Parse an explicit boolean environment-variable override."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{variable_name} must be one of true/false, 1/0, yes/no, or on/off; got {value!r}."
    )


def _parse_positive_int_environment(value: str, variable_name: str) -> int:
    """Parse a positive integer environment override."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{variable_name} must be a positive integer; got {value!r}.") from exc
    if parsed < 1:
        raise ValueError(f"{variable_name} must be a positive integer; got {value!r}.")
    return parsed


def string_to_uuid(input_string: str) -> str:
    """Convert string to deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(input_string)))


def get_network_info() -> str:
    """Get hostname for identification."""
    hostname = socket.gethostname()
    try:
        all_ips = socket.gethostbyname_ex(hostname)[2]
        real_ips = [ip for ip in all_ips if not ip.startswith("127.")]
        return hostname + (real_ips[0] if real_ips else "")
    except:
        return hostname


class FileSystemReader:
    """Simple local file system reader."""
    
    def __init__(self, **kwargs):
        pass
    
    def read_file(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            return f.read()
    
    def write_file(self, file_path: str, content: Any, append: bool = False) -> bool:
        if isinstance(content, str):
            content = content.encode('utf-8')
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'ab' if append else 'wb') as f:
            f.write(content)
        return True
    
    def exists(self, file_path: str) -> bool:
        return os.path.exists(file_path) and not os.path.isdir(file_path)


class MessageClient:
    """Task submission client for web search."""
    
    def __init__(self, path: str = './cache/task_queue', **kwargs):
        self.path = path
        self._handler = None

    def _get_handler(self):
        """Create the search handler once and expose its resolved config."""
        if self._handler is None:
            from tools_server.handler import Handler
            self._handler = Handler(self._load_config())
        return self._handler
    
    def submit_tasks(self, task_list: List[Dict]) -> List[Dict]:
        """Submit tasks and get results."""
        if not task_list:
            return task_list
        
        try:
            return self._get_handler().handle_all(task_list)
        except Exception as e:
            print(f"[MessageClient] Error: {e}")
            traceback.print_exc()
            for task in task_list:
                if 'content' not in task:
                    task['content'] = f"Error: {str(e)}"
            return task_list
    
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Fetch normalized local-retriever embeddings for a batch of queries.

        Embeddings are rollout metadata, never part of the tool observation
        serialized into the policy context.
        """
        if not queries:
            return []
        if not all(isinstance(query, str) and query.strip() for query in queries):
            raise ValueError("embed_queries requires non-empty query strings")

        config = self._get_handler().config
        retrieve_url = str(config.get('local_retriever_url', 'http://127.0.0.1:8002/retrieve')).rstrip('/')
        embed_url = str(config.get('local_retriever_embed_url', '')).strip()
        if not embed_url:
            embed_url = retrieve_url[:-len('/retrieve')] + '/embed' if retrieve_url.endswith('/retrieve') else retrieve_url + '/embed'

        timeout = float(config.get('local_retriever_timeout', 30))
        response = requests.post(embed_url, json={"queries": queries}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get('embeddings') if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(queries):
            actual_length = len(embeddings) if isinstance(embeddings, list) else 'n/a'
            raise RuntimeError(
                "Local retriever /embed returned a mismatched response: "
                f"expected {len(queries)} embeddings, got {type(embeddings).__name__} with length {actual_length}"
            )
        return embeddings

    def _load_config(self) -> Dict:
        """Load config from YAML and apply explicit environment overrides."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        try:
            import yaml
        except ImportError:
            config = {}
        else:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError):
                config = {}

        if not isinstance(config, dict):
            config = {}

        if not config:
            config = {
                'search_engine': 'local_retriever',
                'search_top_k': 10,
                'local_retriever_url': 'http://127.0.0.1:8002/retrieve',
                'local_retriever_embed_url': 'http://127.0.0.1:8002/embed',
                'cache_dir': './cache/tool_cache',
            }

        mock_search = os.getenv('IGPO_MOCK_SEARCH')
        if mock_search is not None:
            config['mock_mode'] = _parse_bool_environment(mock_search, 'IGPO_MOCK_SEARCH')

        # Experiment-scoped overrides avoid editing config.yaml for every
        # single-query/top-k retrieval ablation.
        search_top_k = os.getenv('IGPO_SEARCH_TOP_K')
        if search_top_k is not None:
            config['search_top_k'] = _parse_positive_int_environment(
                search_top_k, 'IGPO_SEARCH_TOP_K'
            )

        max_search_queries = os.getenv('IGPO_MAX_SEARCH_QUERIES')
        if max_search_queries is not None:
            config['max_search_queries'] = _parse_positive_int_environment(
                max_search_queries, 'IGPO_MAX_SEARCH_QUERIES'
            )

        return config
