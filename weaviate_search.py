import argparse
import difflib
import functools
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from weaviate.classes.query import Filter, MetadataQuery, Rerank

from weaviate_utils import digits_only, get_rerank_score, make_weaviate_client


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
    query_properties: list[str] | None = None,
    rerank_prop: str = "text",
    filters: Any | None = None,
) -> dict[str, Any] | None:
    if not client.collections.exists(collection_name):
        _eprint(
            f"Collection {collection_name!r} not found in Weaviate schema. "
            "Run `weaviate_index.py --recreate` to (re)create collections."
        )
        return None
    collection = client.collections.get(collection_name)
    try:
        resp = collection.query.hybrid(
            query=query,
            alpha=alpha,
            query_properties=query_properties or ["text"],
            limit=top_k,
            filters=filters,
            rerank=Rerank(prop=rerank_prop, query=query),
            return_metadata=MetadataQuery(score=True),
        )
    except Exception as e:  # noqa: BLE001
        _eprint(f"{collection_name}: query failed: {e}")
        return None
    objs = list(getattr(resp, "objects", []) or [])
    if not objs:
        return None

    window = len(objs)
    if rerank_top_n is not None and rerank_top_n > 0:
        window = min(window, int(rerank_top_n))
    candidates = objs[:window]

    best_obj = None
    best_score = None
    for obj in candidates:
        score = get_rerank_score(obj)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_obj = obj
    if best_obj is None:
        best_obj = candidates[0]
        best_score = get_rerank_score(best_obj)

    props = dict(getattr(best_obj, "properties", {}) or {})
    if best_score is not None:
        props["score"] = best_score
    return props


def hs6_from_country_match(
    client: Any,
    *,
    hs_country: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not hs_country:
        return None
    if not client.collections.exists("HS6"):
        return None
    digits = digits_only(str(hs_country.get("hs_code") or ""))
    if len(digits) < 6:
        return None
    prefix6 = digits[:6]
    collection = client.collections.get("HS6")
    resp = collection.query.fetch_objects(filters=Filter.by_property("digits").equal(prefix6), limit=1)
    objs = list(getattr(resp, "objects", []) or [])
    if not objs:
        return None
    props = dict(getattr(objs[0], "properties", {}) or {})
    props["score"] = hs_country.get("score")
    return props


def keep_if_score_at_least(item: dict[str, Any] | None, min_score: float | None) -> dict[str, Any] | None:
    if item is None or min_score is None:
        return item
    score = item.get("score")
    if score is None:
        return None
    try:
        score_val = float(score)
    except Exception:  # noqa: BLE001
        return None
    return item if score_val >= min_score else None


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_QUERY_IGNORE_TOKENS = {
    "hs",
    "hs6",
    "hs8",
    "hs10",
    "hs12",
    "hts",
    "tariff",
    "code",
    "codes",
    "classification",
    "classify",
    "search",
    "find",
}


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in _TOKEN_RE.findall(text or "") if t}


def _query_tokens(query: str) -> set[str]:
    return {t for t in _tokens(query) if t not in _QUERY_IGNORE_TOKENS}


def adaptive_hs_top_k(query: str, base_top_k: int) -> int:
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return base_top_k
    if all(t.isdigit() for t in q_tokens):
        return base_top_k
    if len(q_tokens) <= 1:
        return max(base_top_k, 75)
    if len(q_tokens) <= 2:
        return max(base_top_k, 50)
    return base_top_k


def _ascii_fold(text: str) -> str:
    # Keep this conservative: only fold combining marks to support common no-diacritics queries
    # (e.g. "tieu den" matching "tiêu đen").
    return "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch))


def _token_variants(token: str) -> set[str]:
    t = (token or "").casefold()
    out = {t} if t else set()
    folded = _ascii_fold(t)
    if folded and folded != t:
        out.add(folded)

    # Very small English plural heuristic to help one-word product queries like "avocado" ↔ "avocados".
    # Only applies to ASCII tokens to avoid corrupting non-Latin scripts.
    for v in list(out):
        if v.isascii() and len(v) > 3 and v.endswith("s") and not v.endswith("ss"):
            out.add(v[:-1])
    return out


def _normalized_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text or ""):
        out |= _token_variants(tok)
    return out


def _hs_overlap_tokens(query: str, item: dict[str, Any]) -> set[str]:
    q = {t for t in _normalized_tokens(query) if t and t not in _QUERY_IGNORE_TOKENS and len(t) >= 3}
    if not q:
        return set()

    keywords = item.get("keywords")
    kw_text = " ".join(str(k) for k in keywords) if isinstance(keywords, list) else ""
    cand = " ".join(
        p
        for p in [
            str(item.get("description") or ""),
            str(item.get("description_raw") or ""),
            kw_text,
        ]
        if p
    )
    c = {t for t in _normalized_tokens(cand) if t and len(t) >= 3}
    return q & c


def _token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if abs(len(a) - len(b)) > 2:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _token_match_threshold(q_tokens: set[str], qt: str, pt: str) -> float:
    if len(q_tokens) != 1:
        return 0.9
    min_len = min(len(qt), len(pt))
    if min_len >= 6:
        return 0.83
    if min_len == 5:
        return 0.85
    if min_len == 4:
        return 0.9
    return 0.9


def _matched_product_tokens(q_tokens: set[str], product_tokens: set[str]) -> set[str]:
    inter = q_tokens & product_tokens
    if inter:
        return inter
    matched: set[str] = set()
    for pt in product_tokens:
        if len(pt) < 4:
            continue
        for qt in q_tokens:
            if len(qt) < 4:
                continue
            if _token_similarity(qt, pt) >= _token_match_threshold(q_tokens, qt, pt):
                matched.add(pt)
                break
    return matched


def _matched_query_tokens(q_tokens: set[str], product_tokens: set[str]) -> set[str]:
    matched: set[str] = set()
    for qt in q_tokens:
        if qt in product_tokens:
            matched.add(qt)
            continue
        if len(qt) < 4:
            continue
        for pt in product_tokens:
            if len(pt) < 4:
                continue
            if _token_similarity(qt, pt) >= _token_match_threshold(q_tokens, qt, pt):
                matched.add(qt)
                break
    return matched


def _fuzzy_overlap_score(query: str, item: dict[str, Any]) -> tuple[float, float]:
    q = {t for t in _normalized_tokens(query) if t and len(t) >= 3 and t not in _QUERY_IGNORE_TOKENS}
    if not q:
        return 0.0, 0.0
    keywords = item.get("keywords")
    kw_text = " ".join(str(k) for k in keywords) if isinstance(keywords, list) else ""
    cand = " ".join(
        p
        for p in [
            str(item.get("description") or ""),
            str(item.get("description_raw") or ""),
            kw_text,
        ]
        if p
    )
    c = {t for t in _normalized_tokens(cand) if t and len(t) >= 3}
    if not c:
        return 0.0, 0.0
    bests: list[float] = []
    for qt in q:
        best = 0.0
        for ct in c:
            score = _token_similarity(qt, ct)
            if score > best:
                best = score
            if best >= 1.0:
                break
        bests.append(best)
    avg = sum(bests) / len(bests) if bests else 0.0
    return avg, max(bests) if bests else 0.0


def fuzzy_hs_thresholds(query: str, base_avg: float, base_max: float) -> tuple[float, float]:
    tokens = [t for t in _normalized_tokens(query) if t and t not in _QUERY_IGNORE_TOKENS]
    if not tokens or any(t.isdigit() for t in tokens):
        return base_avg, base_max
    if len(tokens) != 1:
        return base_avg, base_max

    token = tokens[0]
    if len(token) <= 3:
        return max(base_avg, 0.95), max(base_max, 0.97)
    if len(token) == 4:
        return max(base_avg, 0.9), max(base_max, 0.93)
    if len(token) == 5:
        return min(base_avg, 0.85), min(base_max, 0.87)
    if len(token) == 6:
        return min(base_avg, 0.82), min(base_max, 0.83)
    return min(base_avg, 0.82), min(base_max, 0.84)


def fuzzy_hs_match(
    client: Any,
    *,
    collection_name: str,
    query: str,
    min_avg: float,
    min_max: float,
    limit: int = 5000,
) -> dict[str, Any] | None:
    if not client.collections.exists(collection_name):
        return None
    collection = client.collections.get(collection_name)
    resp = collection.query.fetch_objects(limit=limit)
    objs = list(getattr(resp, "objects", []) or [])
    if not objs:
        return None
    best_props = None
    best_avg = 0.0
    best_max = 0.0
    for obj in objs:
        props = dict(getattr(obj, "properties", {}) or {})
        avg, mx = _fuzzy_overlap_score(query, props)
        if avg > best_avg or (avg == best_avg and mx > best_max):
            best_avg = avg
            best_max = mx
            best_props = props
    if best_props is None:
        return None
    if best_avg >= min_avg and best_max >= min_max:
        best_props["score"] = max(best_avg, best_max)
        return best_props
    return None


def keep_hs_if_confident(
    item: dict[str, Any] | None,
    min_score: float | None,
    *,
    query: str,
) -> dict[str, Any] | None:
    """
    HS rerank scores can be negative even for correct matches (they are model logits),
    so only apply the score threshold when there is no lexical overlap between the
    query and the item's description/keywords.
    """

    if item is None or min_score is None:
        return item

    if _hs_overlap_tokens(query, item):
        return item

    return keep_if_score_at_least(item, min_score)


@functools.lru_cache(maxsize=1)
def _product_tokens_by_product() -> dict[str, set[str]]:
    """
    Loads multilingual product keywords (if present) so we can suppress attribute triples
    for product-only queries (e.g. "black pepper", "tiêu đen") without any LLM at search time.
    """

    cache_path = Path(__file__).resolve().parent / ".cache" / "attrs_augment" / "augmented_products.jsonl"
    if not cache_path.exists():
        return {}

    out: dict[str, set[str]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(row, dict):
            continue
        product = row.get("id")
        if not isinstance(product, str) or not product.strip():
            continue
        kws = row.get("keywords")
        terms: list[str] = []
        if isinstance(kws, list):
            terms.extend([str(k) for k in kws if str(k).strip()])
        terms.append(product)

        tok: set[str] = set()
        for t in terms:
            tok |= _tokens(t)
        if tok:
            out[product] = tok

    return out


@functools.lru_cache(maxsize=1)
def _product_token_idf() -> dict[str, float]:
    products = _product_tokens_by_product()
    if not products:
        return {}
    n = len(products)
    df: dict[str, int] = {}
    for toks in products.values():
        for t in toks:
            df[t] = df.get(t, 0) + 1
    # Smooth IDF; always > 0, higher => more distinguishing across products.
    return {t: (math.log((n + 1) / (c + 1)) + 1.0) for t, c in df.items()}


def _best_product_match(q_tokens: set[str]) -> tuple[str, float] | None:
    products = _product_tokens_by_product()
    if not products or not q_tokens:
        return None
    idf = _product_token_idf()
    if not idf:
        return None

    best_product: str | None = None
    best_weight = 0.0
    best_max_idf = 0.0
    second_weight = 0.0

    for product, ptoks in products.items():
        inter = _matched_product_tokens(q_tokens, ptoks)
        if not inter:
            continue
        w = sum(idf.get(t, 1.0) for t in inter)
        mx = max(idf.get(t, 1.0) for t in inter)
        if w > best_weight:
            second_weight = best_weight
            best_product = product
            best_weight = w
            best_max_idf = mx
        elif w > second_weight:
            second_weight = w

    if best_product is None or best_weight <= 0:
        return None

    # Ambiguity guard: require a distinguishing token OR a clear separation from runner-up.
    if best_max_idf < 1.4 and (best_weight - second_weight) < 0.75:
        return None

    return best_product, best_weight


def hybrid_search_best_attribute(
    client: Any,
    *,
    query: str,
    alpha: float,
    top_k: int,
    rerank_top_n: int,
    min_score: float | None,
    query_properties: list[str] | None = None,
    rerank_prop: str = "text",
) -> dict[str, Any] | None:
    """
    Returns the best attribute triple, but only when the query actually contains an
    attribute signal beyond the product name.

    Example: "black pepper" -> attribute=None (avoid redundant "Variety: Black")
             "organic black pepper" -> attribute=... if present.
    """

    q_tokens = _query_tokens(query)
    if not q_tokens:
        return None

    collection = client.collections.get("Attributes")
    # Consider a small rerank window for gating; keep at least 1.
    window = max(1, int(rerank_top_n))
    product_tokens = _product_tokens_by_product()

    # Step 1: Resolve the most likely product label from the multilingual product lexicon.
    # This prevents cases like "white pepper" incorrectly returning "Black Pepper: Variety : White"
    # by preferring the canonical product "White Pepper" when it exists.
    best_prod = _best_product_match(q_tokens)
    if best_prod is not None:
        product, prod_weight = best_prod
        prod_tokens = product_tokens.get(product) or _tokens(product)
        matched = _matched_query_tokens(q_tokens, prod_tokens or set())
        leftover = q_tokens - matched

        # If the query is product-only (no extra signal beyond product), return just the product name.
        # Use the product-match weight (IDF-weighted token overlap) as the confidence score for the
        # "product-only" result (instead of the reranker score), so a query like "black pepper" doesn't
        # get a misleadingly high score from a redundant attribute triple.
        if not leftover:
            out = {"product": product, "score": prod_weight}
            return out if keep_if_score_at_least(out, min_score) is not None else None

        resp = collection.query.hybrid(
            query=query,
            alpha=alpha,
            query_properties=query_properties or ["text"],
            limit=top_k,
            filters=Filter.by_property("product").equal(product),
            rerank=Rerank(prop=rerank_prop, query=query),
            return_metadata=MetadataQuery(score=True),
        )
        objs = list(getattr(resp, "objects", []) or [])

        # Otherwise, find the best attribute for that product whose cues match the leftover tokens.
        for obj in objs[:window]:
            props = dict(getattr(obj, "properties", {}) or {})
            score = get_rerank_score(obj)
            if score is not None:
                props["score"] = score
            if keep_if_score_at_least(props, min_score) is None:
                continue

            cue_tokens: set[str] = set()
            cue_tokens |= _tokens(str(props.get("attribute_type") or ""))
            cue_tokens |= _tokens(str(props.get("attribute_value") or ""))
            kws = props.get("keywords")
            if isinstance(kws, list):
                for k in kws:
                    cue_tokens |= _tokens(str(k))
            if cue_tokens and not (leftover & cue_tokens):
                continue

            return props

        # No attribute-specific match: return the product-only result (still helpful in UI).
        out = {"product": product, "score": prod_weight}
        return out if keep_if_score_at_least(out, min_score) is not None else None

    # Step 2 (fallback): Attribute-driven matching when we can't confidently resolve a product label.
    resp = collection.query.hybrid(
        query=query,
        alpha=alpha,
        query_properties=query_properties or ["text"],
        limit=top_k,
        rerank=Rerank(prop=rerank_prop, query=query),
        return_metadata=MetadataQuery(score=True),
    )
    objs = list(getattr(resp, "objects", []) or [])
    if not objs:
        return None

    best_product_only: dict[str, Any] | None = None

    for obj in objs[:window]:
        props = dict(getattr(obj, "properties", {}) or {})
        score = get_rerank_score(obj)
        if score is not None:
            props["score"] = score
        if keep_if_score_at_least(props, min_score) is None:
            continue

        product = str(props.get("product") or "").strip()
        if not product:
            continue
        prod_tokens = product_tokens.get(product) or _tokens(product)
        if not prod_tokens:
            continue

        # Query must mention the product (in any language we have keywords for).
        matched = _matched_query_tokens(q_tokens, prod_tokens)
        product_overlap = matched

        # If the query contains only product tokens, suppress attribute output.
        leftover = q_tokens - matched if product_overlap else set(q_tokens)

        if product_overlap and not leftover:
            # Product-only query: return product (no attribute_type/value) instead of a redundant triple.
            if best_product_only is None:
                best_product_only = {"product": product, "score": props.get("score")}
            continue

        cue_tokens: set[str] = set()
        type_tokens = _tokens(str(props.get("attribute_type") or ""))
        value_tokens = _tokens(str(props.get("attribute_value") or ""))
        cue_tokens |= type_tokens
        cue_tokens |= value_tokens
        kws = props.get("keywords")
        if isinstance(kws, list):
            for k in kws:
                cue_tokens |= _tokens(str(k))

        # If the product wasn't mentioned, still allow an "attribute-only" match when the
        # query clearly matches the attribute type/value itself (e.g. "ceylon").
        if not product_overlap:
            if (q_tokens & (type_tokens | value_tokens)) and (q_tokens & cue_tokens):
                return props
            continue

        if cue_tokens and not (leftover & cue_tokens):
            continue

        return props

    return best_product_only


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
    parser.add_argument(
        "--min-attribute-score",
        type=float,
        default=0.0,
        help="Drop the attribute match if its score is below this value (default: 0.0).",
    )
    parser.add_argument(
        "--min-hs6-score",
        type=float,
        default=0.0,
        help="Drop the 6-digit HS match if its score is below this value (default: 0.0).",
    )
    parser.add_argument(
        "--min-hs-country-score",
        type=float,
        default=0.0,
        help="Drop the country-specific HS match if its score is below this value (default: 0.0).",
    )
    parser.add_argument(
        "--fuzzy-hs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fuzzy fallback for HS lookups (default: True).",
    )
    parser.add_argument(
        "--fuzzy-hs-min-avg",
        type=float,
        default=0.84,
        help="Base average fuzzy token similarity for HS fallback (default: 0.84).",
    )
    parser.add_argument(
        "--fuzzy-hs-min-max",
        type=float,
        default=0.9,
        help="Base best-token similarity for HS fallback (default: 0.9).",
    )
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
        hs_query_props = {
            "HS6": ["text", "hs_code", "digits", "description", "description_raw", "keywords"],
            "HSCountry": [
                "text",
                "country",
                "hs_code",
                "digits",
                "description",
                "description_raw",
                "keywords",
            ],
            "Attributes": ["text", "product", "attribute_type", "attribute_value", "keywords"],
        }

        hs_top_k = adaptive_hs_top_k(q, args.top_k)
        hs_rerank_top_n = max(args.rerank_top_n, hs_top_k)
        fuzzy_avg, fuzzy_max = fuzzy_hs_thresholds(q, args.fuzzy_hs_min_avg, args.fuzzy_hs_min_max)

        hs_country_raw = hybrid_search_top1(
            client,
            collection_name="HSCountry",
            query=q,
            alpha=args.alpha,
            top_k=hs_top_k,
            rerank_top_n=hs_rerank_top_n,
            query_properties=hs_query_props["HSCountry"],
        )
        hs_country = keep_hs_if_confident(hs_country_raw, args.min_hs_country_score, query=q)
        if hs_country is None and args.fuzzy_hs:
            hs_country = fuzzy_hs_match(
                client,
                collection_name="HSCountry",
                query=q,
                min_avg=fuzzy_avg,
                min_max=fuzzy_max,
            )

        hs6 = hs6_from_country_match(client, hs_country=hs_country)
        if hs6 is None:
            hs6 = hybrid_search_top1(
                client,
                collection_name="HS6",
                query=q,
                alpha=args.alpha,
                top_k=hs_top_k,
                rerank_top_n=hs_rerank_top_n,
                query_properties=hs_query_props["HS6"],
            )
        hs6 = keep_hs_if_confident(hs6, args.min_hs6_score, query=q)
        if hs6 is None and args.fuzzy_hs:
            hs6 = fuzzy_hs_match(
                client,
                collection_name="HS6",
                query=q,
                min_avg=fuzzy_avg,
                min_max=fuzzy_max,
            )

        attr_raw = hybrid_search_best_attribute(
            client,
            query=q,
            alpha=args.alpha,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
            min_score=args.min_attribute_score,
            query_properties=hs_query_props["Attributes"],
        )
        attr = attr_raw
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
        if attr.get("attribute_type") and attr.get("attribute_value"):
            triple = f"{attr.get('product')}: {attr.get('attribute_type')} : {attr.get('attribute_value')}"
        else:
            triple = str(attr.get("product") or "")
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
