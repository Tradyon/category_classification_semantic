import argparse
import json
import sys
from typing import Any

from weaviate.classes.query import MetadataQuery, Rerank

from weaviate_utils import get_rerank_score, make_weaviate_client


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def hybrid_search_top1(
    client: Any,
    *,
    collection_name: str,
    query: str,
    alpha: float,
    top_k: int,
    rerank_top_n: int,
) -> dict[str, Any] | None:
    collection = client.collections.get(collection_name)
    resp = collection.query.hybrid(
        query=query,
        alpha=alpha,
        query_properties=["text"],
        limit=top_k,
        rerank=Rerank(prop="text", query=query),
        return_metadata=MetadataQuery(score=True),
    )
    objs = list(getattr(resp, "objects", []) or [])
    if not objs:
        return None
    obj0 = objs[0] if rerank_top_n <= 1 else objs[:rerank_top_n][0]
    props = dict(getattr(obj0, "properties", {}) or {})
    score = get_rerank_score(obj0)
    if score is not None:
        props["score"] = score
    return props


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid + reranked search over HS codes and attribute triples stored in Weaviate."
    )
    parser.add_argument("query", type=str, help="Free-text search query")
    parser.add_argument("--weaviate-url", type=str, default=None)
    parser.add_argument("--weaviate-api-key", type=str, default=None)
    parser.add_argument("--grpc-port", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha (0=BM25, 1=vector)")
    parser.add_argument("--top-k", type=int, default=20, help="Initial hybrid candidate count")
    parser.add_argument("--rerank-top-n", type=int, default=5, help="Rerank window")
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    q = args.query.strip()
    if not q:
        raise SystemExit("query must be non-empty")

    client = make_weaviate_client(
        weaviate_url=args.weaviate_url,
        weaviate_api_key=args.weaviate_api_key,
        grpc_port=args.grpc_port,
    )
    try:
        hs6 = hybrid_search_top1(
            client,
            collection_name="HS6",
            query=q,
            alpha=args.alpha,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
        )
        hs_country = hybrid_search_top1(
            client,
            collection_name="HSCountry",
            query=q,
            alpha=args.alpha,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
        )
        attr = hybrid_search_top1(
            client,
            collection_name="Attributes",
            query=q,
            alpha=args.alpha,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
        )
    finally:
        client.close()

    out: dict[str, Any] = {"query": q, "hs6": None, "hs_country": None, "attribute": None}
    if hs6:
        out["hs6"] = {
            "hs_code": hs6.get("hs_code"),
            "description": hs6.get("description"),
            "score": hs6.get("score"),
        }
    if hs_country:
        out["hs_country"] = {
            "country": hs_country.get("country"),
            "hs_code": hs_country.get("hs_code"),
            "description": hs_country.get("description"),
            "score": hs_country.get("score"),
        }
    if attr:
        triple = f"{attr.get('product')}: {attr.get('attribute_type')} : {attr.get('attribute_value')}"
        out["attribute"] = {
            "triple": triple,
            "product": attr.get("product"),
            "attribute_type": attr.get("attribute_type"),
            "attribute_value": attr.get("attribute_value"),
            "score": attr.get("score"),
        }

    if args.pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

