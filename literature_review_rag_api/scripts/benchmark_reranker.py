#!/usr/bin/env python
"""Benchmark reranker latency impact on search queries.

Compares query latency with reranking enabled vs disabled.
"""

import argparse
import json
import statistics
import time
from typing import Dict, Any, List

from literature_rag.config import load_config
from literature_rag.literature_rag import LiteratureReviewRAG


# Sample queries for benchmarking
BENCHMARK_QUERIES = [
    "business formation in the Ruhr region",
    "institutional economics and regional development",
    "deindustrialization effects on labor markets",
    "COVID-19 impact on German economy",
    "varieties of capitalism comparative analysis",
    "spatial panel data methods for regional analysis",
    "municipal strategy documents urban planning",
    "energy transition just transition policies",
]


def create_rag_instance(config, use_reranking: bool) -> LiteratureReviewRAG:
    """Create a RAG instance with specified reranking setting."""
    return LiteratureReviewRAG(
        chroma_path=config.storage.indices_path,
        config={
            "device": config.embedding.device,
            "collection_name": config.storage.collection_name,
            "expand_queries": config.retrieval.expand_queries,
            "max_expansions": config.retrieval.max_expansions,
            "use_reranking": use_reranking,
            "reranker_model": config.retrieval.reranker_model,
            "rerank_top_k": config.retrieval.rerank_top_k,
            "term_maps": config.normalization.term_maps if config.normalization.enable else {}
        },
        embedding_model=config.embedding.model
    )


def benchmark_queries(
    rag: LiteratureReviewRAG,
    queries: List[str],
    n_results: int,
    warmup_runs: int = 2
) -> Dict[str, Any]:
    """Run benchmark queries and collect timing statistics."""

    # Warmup runs (not counted)
    print(f"  Running {warmup_runs} warmup queries...")
    for i in range(warmup_runs):
        rag.query(question=queries[i % len(queries)], n_results=n_results)

    # Actual benchmark
    latencies = []
    results_data = []

    print(f"  Running {len(queries)} benchmark queries...")
    for query in queries:
        start = time.perf_counter()
        result = rag.query(question=query, n_results=n_results)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)
        results_data.append({
            "query": query,
            "latency_ms": round(elapsed_ms, 2),
            "num_results": len(result.get("documents", [[]])[0])
        })

    return {
        "latencies_ms": latencies,
        "results": results_data,
        "stats": {
            "count": len(latencies),
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) >= 20 else None,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark reranker latency")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results per query")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--queries", type=str, default=None, help="Optional JSON file with custom queries")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()

    # Load queries
    if args.queries:
        with open(args.queries) as f:
            queries = json.load(f)
    else:
        queries = BENCHMARK_QUERIES

    print(f"Benchmarking with {len(queries)} queries, n_results={args.n_results}")
    print()

    config = load_config()

    # Benchmark WITHOUT reranking
    print("=" * 60)
    print("BENCHMARK: Reranking DISABLED")
    print("=" * 60)
    rag_no_rerank = create_rag_instance(config, use_reranking=False)
    results_no_rerank = benchmark_queries(rag_no_rerank, queries, args.n_results, args.warmup)
    print(f"  Mean latency: {results_no_rerank['stats']['mean_ms']} ms")
    print(f"  Median latency: {results_no_rerank['stats']['median_ms']} ms")
    print(f"  Min/Max: {results_no_rerank['stats']['min_ms']} / {results_no_rerank['stats']['max_ms']} ms")
    print()

    # Benchmark WITH reranking
    print("=" * 60)
    print("BENCHMARK: Reranking ENABLED")
    print(f"  Model: {config.retrieval.reranker_model}")
    print(f"  Top-K candidates: {config.retrieval.rerank_top_k}")
    print("=" * 60)
    rag_rerank = create_rag_instance(config, use_reranking=True)

    # First query with reranking will be slow (model loading)
    print("  Loading reranker model (first query)...")
    start = time.perf_counter()
    rag_rerank.query(question="test query", n_results=args.n_results)
    model_load_time = (time.perf_counter() - start) * 1000
    print(f"  Model load time: {model_load_time:.0f} ms")

    results_rerank = benchmark_queries(rag_rerank, queries, args.n_results, args.warmup)
    print(f"  Mean latency: {results_rerank['stats']['mean_ms']} ms")
    print(f"  Median latency: {results_rerank['stats']['median_ms']} ms")
    print(f"  Min/Max: {results_rerank['stats']['min_ms']} / {results_rerank['stats']['max_ms']} ms")
    print()

    # Comparison
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    overhead_mean = results_rerank['stats']['mean_ms'] - results_no_rerank['stats']['mean_ms']
    overhead_pct = (overhead_mean / results_no_rerank['stats']['mean_ms']) * 100
    print(f"  Reranking overhead (mean): +{overhead_mean:.0f} ms (+{overhead_pct:.0f}%)")
    print()

    # Recommendation
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if overhead_mean < 100:
        print("  Reranking adds minimal overhead (<100ms). Safe to enable.")
    elif overhead_mean < 300:
        print("  Reranking adds moderate overhead (100-300ms). Enable for quality-critical use cases.")
    else:
        print("  Reranking adds significant overhead (>300ms). Consider async reranking or caching.")
    print()

    # Save results
    report = {
        "config": {
            "n_results": args.n_results,
            "reranker_model": config.retrieval.reranker_model,
            "rerank_top_k": config.retrieval.rerank_top_k,
            "num_queries": len(queries)
        },
        "without_reranking": results_no_rerank,
        "with_reranking": results_rerank,
        "overhead": {
            "mean_ms": round(overhead_mean, 2),
            "percentage": round(overhead_pct, 2)
        }
    }

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to {args.out}")
    else:
        print("Full results:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
