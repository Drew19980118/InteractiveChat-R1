#!/usr/bin/env python3
"""Serve a local FAISS dense retriever for InteractiveChat-R1.

The API is intentionally compatible with the retriever client used by
``tools_server.search.search_api``::

    POST /retrieve
    {"queries": ["..."], "topk": 3, "return_scores": true}

    {"result": [[{"document": {...}, "score": 0.0}]]}

The service loads its FAISS index, corpus, and encoder once at startup.
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

# Reserve separate GPUs for policy training and the frozen user simulator.
# independent retriever process, while allowing an explicit shell setting to
# take precedence.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field



def load_corpus(corpus_path: str):
    """Load the JSONL corpus once and provide random access by FAISS id."""
    import datasets
    return datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        num_proc=4,
    )


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None].clamp_min(1)


class Encoder:
    """E5/BGE-compatible Transformer query encoder from the supplied script."""

    def __init__(self, model_name: str, model_path: str, max_length: int, use_fp16: bool):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        # Load the config explicitly so remote-code model repositories retain
        # the same behavior as the supplied script.
        AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda")
        self.model.to(self.device).eval()
        if use_fp16:
            self.model.half()

    @torch.inference_mode()
    def encode(self, query_list: List[str]) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        model_name = self.model_name.lower()
        if "e5" in model_name:
            query_list = [f"query: {query}" for query in query_list]
        elif "bge" in model_name:
            query_list = [
                f"Represent this sentence for searching relevant passages: {query}"
                for query in query_list
            ]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        embeddings = mean_pooling(output.last_hidden_state, inputs["attention_mask"])
        if "dpr" not in model_name:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings.float().cpu().numpy().astype(np.float32, order="C")


class DenseRetriever:
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        model_name: str,
        model_path: str,
        default_topk: int,
        batch_size: int,
        max_length: int,
        use_fp16: bool,
        use_gpu_index: bool,
    ):
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.isfile(corpus_path):
            raise FileNotFoundError(f"JSONL corpus not found: {corpus_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this local retriever but is not available.")

        import faiss

        print(f"Loading FAISS index: {index_path}", flush=True)
        self.index = faiss.read_index(index_path)
        if use_gpu_index:
            clone_options = faiss.GpuMultipleClonerOptions()
            clone_options.useFloat16 = True
            clone_options.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, clone_options)

        print(f"Loading corpus: {corpus_path}", flush=True)
        self.corpus = load_corpus(corpus_path)
        self.default_topk = default_topk
        self.batch_size = batch_size
        self.encoder = Encoder(
            model_name=model_name,
            model_path=model_path,
            max_length=max_length,
            use_fp16=use_fp16,
        )

    def batch_search(
        self,
        query_list: List[str],
        topk: Optional[int] = None,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[float]]]:
        topk = topk or self.default_topk
        if topk < 1:
            raise ValueError("topk must be at least 1")

        all_documents: List[List[Dict[str, Any]]] = []
        all_scores: List[List[float]] = []
        for start in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start:start + self.batch_size]
            query_embeddings = self.encoder.encode(query_batch)
            scores, indices = self.index.search(query_embeddings, k=topk)

            for row_indices, row_scores in zip(indices.tolist(), scores.tolist()):
                documents = []
                valid_scores = []
                for document_index, score in zip(row_indices, row_scores):
                    # FAISS emits -1 when fewer than topk documents exist.
                    if document_index < 0:
                        continue
                    documents.append(dict(self.corpus[int(document_index)]))
                    valid_scores.append(float(score))
                all_documents.append(documents)
                all_scores.append(valid_scores)

        return all_documents, all_scores

    def embed_queries(self, query_list: List[str]) -> np.ndarray:
        """Encode queries with the already-loaded retrieval encoder.

        The returned rows are L2-normalized for E5/BGE, so their dot product is
        cosine similarity. Keep the same chunking as retrieval to bound memory
        use for a large rollout group.
        """
        if not query_list:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = []
        for start in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start:start + self.batch_size]
            embeddings.append(self.encoder.encode(query_batch))

        result = np.concatenate(embeddings, axis=0)
        if result.ndim != 2 or result.shape[0] != len(query_list):
            raise RuntimeError(
                "Local retriever encoder returned an invalid embedding matrix: "
                f"shape={result.shape}, query_count={len(query_list)}"
            )
        return result


class QueryRequest(BaseModel):
    queries: List[str] = Field(min_length=1)
    topk: Optional[int] = Field(default=None, ge=1)
    return_scores: bool = False


class EmbedRequest(BaseModel):
    """Optional batch query embedding endpoint for compatible clients."""

    queries: List[str] = Field(min_length=1)


app = FastAPI(title="InteractiveChat-R1 Local Retriever")
retriever: Optional[DenseRetriever] = None


@app.get("/health")
def health() -> Dict[str, str]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized")
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> Dict[str, List]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized")

    results, scores = retriever.batch_search(request.queries, request.topk)
    if request.return_scores:
        return {
            "result": [
                [
                    {"document": document, "score": score}
                    for document, score in zip(query_results, query_scores)
                ]
                for query_results, query_scores in zip(results, scores)
            ]
        }
    return {"result": results}


@app.post("/embed")
def embed_endpoint(request: EmbedRequest) -> Dict[str, List[List[float]]]:
    """Return normalized E5/BGE query embeddings without retrieval."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized")

    embeddings = np.asarray(retriever.embed_queries(request.queries), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(request.queries):
        raise RuntimeError(
            "Retriever /embed returned an invalid embedding matrix: "
            f"shape={embeddings.shape}, query_count={len(request.queries)}"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3):
        raise RuntimeError("Retriever /embed must return finite L2-normalized embeddings")
    return {"embeddings": embeddings.tolist()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local FAISS retriever for InteractiveChat-R1.")
    parser.add_argument(
        "--index_path",
        default="./collection/qrecc/e5_Flat.index",
        help="Single FAISS index file read by faiss.read_index().",
    )
    parser.add_argument(
        "--corpus_path",
        default="./collection/qrecc/qrecc_index.jsonl",
        help="JSONL corpus whose row order matches the FAISS ids.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Default number of passages per query.")
    parser.add_argument("--retriever_name", default="e5", help="Encoder family, e.g. e5 or bge.")
    parser.add_argument(
        "--retriever_model",
        default="intfloat/e5-base-v2",
        help="Local path or Hugging Face model id for the query encoder.",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--faiss_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place the FAISS index on all GPUs visible to this process.",
    )
    return parser.parse_args()


def main() -> None:
    global retriever
    args = parse_args()
    print(f"Retriever CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    retriever = DenseRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_name=args.retriever_name,
        model_path=args.retriever_model,
        default_topk=args.topk,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=True,
        use_gpu_index=args.faiss_gpu,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
