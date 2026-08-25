#!/usr/bin/env python3
"""Small client for checking local retriever /retrieve and /embed responses."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


def show_document(rank: int, item: dict[str, Any]) -> None:
    document = item.get("document", item)
    passage_id = document.get("passage_id", "<missing>")
    title = document.get("title", "")
    text = document.get("passage_text", document.get("text", ""))
    score = item.get("score", item.get("scores", "<not returned>"))

    print(f"  [{rank}] passage_id: {passage_id}")
    print(f"      title: {title}")
    print(f"      score: {score}")
    print(f"      text : {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the local InteractiveChat-R1 retriever.")
    parser.add_argument("--url", default="http://127.0.0.1:8002/retrieve")
    parser.add_argument(
        "--check-embed",
        action="store_true",
        help="Also call the sibling /embed endpoint and verify normalized vectors.",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to send. Repeat --query for multiple queries.",
    )
    args = parser.parse_args()

    queries = args.queries or ["What is the capital of France?"]
    payload = {"queries": queries, "topk": args.topk, "return_scores": True}

    print(f"POST {args.url}")
    print(f"queries={queries!r}, topk={args.topk}")
    try:
        response = requests.post(args.url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as exc:
        print(f"Response was not JSON: {exc}\n{response.text}", file=sys.stderr)
        raise SystemExit(1)

    results = data.get("result")
    if not isinstance(results, list):
        print("Unexpected response (missing list field 'result'):")
        print(data)
        raise SystemExit(2)

    for query, retrieved in zip(queries, results):
        print(f"\nQuery: {query}")
        if not retrieved:
            print("  No documents returned.")
            continue
        for rank, item in enumerate(retrieved, start=1):
            show_document(rank, item)

    if args.check_embed:
        embed_url = args.url.rstrip("/")
        embed_url = embed_url[: -len("/retrieve")] + "/embed" if embed_url.endswith("/retrieve") else embed_url + "/embed"
        print(f"\nPOST {embed_url}")
        try:
            embed_response = requests.post(embed_url, json={"queries": queries}, timeout=args.timeout)
            embed_response.raise_for_status()
            embeddings = embed_response.json().get("embeddings")
        except (requests.RequestException, ValueError) as exc:
            print(f"Embedding request failed: {exc}", file=sys.stderr)
            raise SystemExit(3)

        if not isinstance(embeddings, list) or len(embeddings) != len(queries):
            print(f"Unexpected /embed response: {embeddings!r}", file=sys.stderr)
            raise SystemExit(4)
        for query, embedding in zip(queries, embeddings):
            try:
                norm = sum(float(value) ** 2 for value in embedding) ** 0.5
            except (TypeError, ValueError):
                print(f"Invalid embedding for {query!r}: {embedding!r}", file=sys.stderr)
                raise SystemExit(5)
            print(f"embed query={query!r}, dim={len(embedding)}, l2_norm={norm:.6f}")
            if abs(norm - 1.0) > 1e-3:
                print("Embedding is not L2-normalized.", file=sys.stderr)
                raise SystemExit(6)


if __name__ == "__main__":
    main()
