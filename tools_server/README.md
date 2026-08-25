# IGPO Tool Server - Web Search

Lightweight web search tool server for IGPO training.

## Quick Start

1. Start the local retriever on the same machine as IGPO. By default it must
   listen on `http://127.0.0.1:8002/retrieve`. The same server also exposes
   `POST /embed` for batched, L2-normalized E5 query vectors used only by
   `algorithm.query_group_advantage=semantic`.

2. Edit `config.yaml` if you use another port or want another number of hits:
```yaml
search_engine: "local_retriever"
local_retriever_url: "http://127.0.0.1:8002/retrieve"
local_retriever_embed_url: "http://127.0.0.1:8002/embed"
search_top_k: 3
```

3. Run IGPO with `IGPO_MOCK_SEARCH=false`; the tool server will be used automatically.

## Configuration

```yaml
# config.yaml
search_engine: "local_retriever"  # or "google" / "bing"
search_top_k: 10            # results per query
local_retriever_url: "http://127.0.0.1:8002/retrieve"
local_retriever_embed_url: "http://127.0.0.1:8002/embed"
```

## Supported Search Engines

| Engine | Provider | Free Tier |
|--------|----------|-----------|
| Google | Serper API | 2,500 searches |
| Bing | Azure | Pay as you go |
| Local retriever | Local QReCC index | No external API |

## Files

```
tools_server/
├── config.yaml      # Configuration
├── handler.py       # Web search handler
├── tools.py         # Tool definition
├── util.py          # MessageClient
└── search/
    └── search_api.py  # Search implementations
```
