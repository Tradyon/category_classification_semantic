import argparse
import csv
import json
import os
import copy
import shutil
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from weaviate.classes.config import Configure, DataType, Property
from weaviate.util import generate_uuid5
from tqdm import tqdm

from weaviate_utils import (
    digits_only,
    infer_country_from_pdf_path,
    is_country_specific_pdf,
    load_env,
    make_weaviate_client,
)

_CACHE_IO_LOCK = threading.Lock()
_STASHED_CACHE_FILES: set[str] = set()

_KEYWORD_LANGS_TARGET = (
    "English (required), Hindi (हिन्दी + Latin transliteration), Urdu (اردو), Arabic (العربية), "
    "French, Russian (русский), Spanish (español), Vietnamese (Tiếng Việt)"
)


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def load_hs_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing HS rows CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"pdf", "page", "country", "hs_code", "description"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"HS CSV missing columns: {sorted(missing)}")
        return [dict(r) for r in reader]


def build_best_desc_by_digits(rows: list[dict[str, str]]) -> dict[str, str]:
    best: dict[str, str] = {}
    for r in rows:
        hs_code = (r.get("hs_code") or "").strip()
        digits = digits_only(hs_code)
        if not digits:
            continue
        desc = _norm_space(str(r.get("description") or ""))
        if not desc:
            continue
        prev = best.get(digits, "")
        if len(desc) > len(prev):
            best[digits] = desc
    return best


def _norm_space(s: str) -> str:
    return " ".join((s or "").strip().split())


def _keyword_allowed_scripts(s: str) -> bool:
    # Allow only Latin (incl. accents), Devanagari, Arabic, and Cyrillic scripts (plus digits/punctuation).
    # This helps keep keywords aligned with the target languages for search.
    for ch in s:
        if ch.isascii():
            continue
        cat = unicodedata.category(ch)
        if not cat or cat[0] not in {"L", "M"}:
            continue
        cp = ord(ch)

        # Combining marks (covers Vietnamese diacritics when decomposed).
        if cat[0] == "M":
            continue

        # Latin (Basic + extended)
        if (
            0x00C0 <= cp <= 0x024F
            or 0x1E00 <= cp <= 0x1EFF
            or 0x2C60 <= cp <= 0x2C7F
            or 0xA720 <= cp <= 0xA7FF
            or 0xAB30 <= cp <= 0xAB6F
        ):
            continue

        # Devanagari
        if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF or 0x11B00 <= cp <= 0x11B5F:
            continue

        # Arabic (+ supplements/presentation forms; covers Urdu)
        if (
            0x0600 <= cp <= 0x06FF
            or 0x0750 <= cp <= 0x077F
            or 0x08A0 <= cp <= 0x08FF
            or 0xFB50 <= cp <= 0xFDFF
            or 0xFE70 <= cp <= 0xFEFF
        ):
            continue

        # Cyrillic
        if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F or 0x1C80 <= cp <= 0x1C8F or 0xA640 <= cp <= 0xA69F:
            continue

        return False
    return True


def _clean_keyword(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = _norm_space(value)
    if not s:
        return None
    if s.lower().startswith("keywords:"):
        s = _norm_space(s.split(":", 1)[-1])
    s = s.strip(" \t\n\r\"'`“”‘’.,;:()[]{}<>|\\/")
    s = _norm_space(s)
    if not s:
        return None
    if len(s) < 2:
        return None
    if not _keyword_allowed_scripts(s):
        return None
    return s


def _dedupe_keywords(values: list[Any], *, limit: int) -> list[str]:
    cleaned: list[str] = []
    for v in values:
        c = _clean_keyword(v)
        if not c:
            continue
        cleaned.append(c)
    out: list[str] = []
    seen: set[str] = set()
    for v in cleaned:
        key = v.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _split_description_segments(desc: str) -> list[str]:
    d = _norm_space(str(desc or ""))
    if not d:
        return []
    d = d.replace("(con.)", " ").replace("(con )", " ")
    parts = [_norm_space(p) for p in d.split(":")]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p:
            continue
        key = p.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= 10:
            break
    return out


def _looks_generic(desc: str) -> bool:
    d = _norm_space(desc).lower().strip(" .:")
    return d in {"other", "other: other", "seed", "other:", "other fruit, fresh. - other"} or d.endswith(":")


def _needs_description_rewrite(desc: str, *, threshold: int) -> bool:
    desc_n = _norm_space(desc)
    if not desc_n:
        return True
    if _looks_generic(desc_n):
        return True
    return len(desc_n) < threshold


def _extract_gemini_usage(resp: Any) -> dict[str, int] | None:
    usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None)
    if usage is None:
        return None

    def pick(*keys: str) -> Any:
        if isinstance(usage, dict):
            for k in keys:
                if k in usage:
                    return usage[k]
            return None
        for k in keys:
            if hasattr(usage, k):
                return getattr(usage, k)
        return None

    prompt = pick("prompt_token_count", "promptTokenCount")
    candidates = pick("candidates_token_count", "candidatesTokenCount")
    total = pick("total_token_count", "totalTokenCount")

    out: dict[str, int] = {}
    if prompt is not None:
        out["prompt_tokens"] = int(prompt)
    if candidates is not None:
        out["completion_tokens"] = int(candidates)
    if total is not None:
        out["total_tokens"] = int(total)
    return out or None


def _finish_reason_name(resp: Any) -> str | None:
    cands = getattr(resp, "candidates", None)
    if not isinstance(cands, list) or not cands:
        return None
    cand0 = cands[0]
    fr = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None)
    if fr is None:
        return None
    name = getattr(fr, "name", None)
    if isinstance(name, str) and name:
        return name
    s = str(fr).strip()
    if not s:
        return None
    if "." in s:
        s = s.split(".")[-1].strip()
    return s or None


def _finish_reason_is_max_tokens(resp: Any) -> bool:
    return (_finish_reason_name(resp) or "").upper() == "MAX_TOKENS"


def _model_requires_property_ordering(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gemini-2.0") or "gemini-2.0" in m


def _add_property_ordering(schema: Any) -> None:
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            schema.setdefault("propertyOrdering", list(props.keys()))
        for v in schema.values():
            _add_property_ordering(v)
        return
    if isinstance(schema, list):
        for v in schema:
            _add_property_ordering(v)


def _add_token_usage(tally: dict[str, int], usage: dict[str, int] | None) -> None:
    if not usage:
        return
    for k, v in usage.items():
        if v is None:
            continue
        tally[k] = int(tally.get(k, 0)) + int(v)


def _load_google_api_key() -> str:
    load_env()
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing `GOOGLE_API_KEY` in environment or `.env` (required for --llm-augment).")
    return key


def _load_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                kw = obj.get("keywords")
                if isinstance(kw, list):
                    obj["keywords"] = _dedupe_keywords(list(kw), limit=12)
                aug_desc = obj.get("augmented_description")
                if isinstance(aug_desc, str):
                    obj["augmented_description"] = _norm_space(aug_desc)
            cache_id = obj.get("id")
            if isinstance(cache_id, str) and cache_id:
                cache[cache_id] = obj
    return cache


def _history_path_for_cache(cache_path: Path) -> Path:
    cache_root = Path(".cache")
    history_root = cache_root / "_history"
    try:
        rel = cache_path.relative_to(cache_root)
    except ValueError:
        rel = Path(cache_path.name)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return history_root / rel.parent / f"{rel.stem}.{ts}.{os.getpid()}{rel.suffix}"


def _dump_jsonl_latest(cache: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for cache_id in sorted(cache.keys()):
        obj = cache.get(cache_id)
        if not isinstance(obj, dict):
            continue
        if obj.get("id") != cache_id:
            obj = dict(obj)
            obj["id"] = cache_id
        lines.append(json.dumps(obj, ensure_ascii=False))
    return ("\n".join(lines) + "\n") if lines else ""


def _write_jsonl_cache_latest(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    new_text = _dump_jsonl_latest(cache)
    resolved = str(cache_path.resolve())

    with _CACHE_IO_LOCK:
        if not new_text and not cache_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing_text = cache_path.read_text(encoding="utf-8") if cache_path.exists() else None
        if existing_text is not None and existing_text == new_text:
            return

        if cache_path.exists() and resolved not in _STASHED_CACHE_FILES:
            hist = _history_path_for_cache(cache_path)
            hist.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, hist)
            _STASHED_CACHE_FILES.add(resolved)

        tmp = cache_path.with_name(cache_path.name + ".tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, cache_path)


def _chunks(seq: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        return [seq]
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _gemini_augment_batch(
    *,
    client: Any,
    model: str,
    items: list[dict[str, Any]],
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from google.genai import types
    from pydantic import BaseModel, Field

    class AugmentedRow(BaseModel):
        id: str = Field(..., description="Stable input id for joining results back to the input.")
        augmented_description: str = Field(
            ...,
            description=(
                "A more descriptive, search-friendly description that stays faithful to the input description. "
                "If rewrite_description=false, return an empty string."
            ),
        )
        keywords: list[str] = Field(
            default_factory=list,
            description=(
                "Short keywords/synonyms to improve search recall, including multiple languages when possible "
                "(max 12 total)."
            ),
        )

    class AugmentedRows(BaseModel):
        rows: list[AugmentedRow]

    prompt = (
        "You improve multilingual search recall for HS code lookups.\n"
        "For each item below, generate search keywords/synonyms (multilingual when possible) and optionally "
        "rewrite the description into a short canonical label.\n"
        "Rules:\n"
        "- Descriptions often contain hierarchical headings separated by punctuation (often ':').\n"
        "  Each item includes `description_segments` (a best-effort split). Use it as a hint, but do not copy the full\n"
        "  hierarchy into the canonical label.\n"
        "- Do NOT introduce goods that are not implied by the input description/context.\n"
        "- If rewrite_description is true: output a concise canonical label (noun phrase) that captures the "
        "core product + only essential qualifiers (e.g. fresh/dried, crushed/ground, organic, color, variety, "
        "greenhouse). Remove legal/heading boilerplate like '(con.)' and long repeated parent headings.\n"
        "- If some segments are generic (e.g. 'Other'), ignore them unless they are the only available signal.\n"
        "- Use context_descriptions to make generic descriptions (e.g. 'Other', 'Seed') self-contained.\n"
        "- If rewrite_description is false, set augmented_description to an empty string.\n"
        "- If rewrite_description is true, keep augmented_description short (<= 120 chars when possible; hard max 220).\n"
        "- Keywords:\n"
        "  - Provide 6-12 short keywords/synonyms as phrases.\n"
        "  - Always include at least 4 English keywords.\n"
        "  - Include trade/common names and scientific names when helpful.\n"
        f"  - Target non-English languages: {_KEYWORD_LANGS_TARGET}.\n"
        "    - Use native scripts (Devanagari for Hindi; Arabic script for Urdu/Arabic; Cyrillic for Russian).\n"
        "    - If unsure about a translation/transliteration, omit it (do not guess).\n"
        "    - Do not include other languages/scripts.\n"
        "  - Include form qualifiers only when they define the HS line (e.g. 'ground pepper' vs 'whole pepper').\n"
        "  - Avoid low-signal words like 'other', 'undamaged' unless essential.\n"
        'Return ONLY JSON object: {"rows":[{id, augmented_description, keywords}, ...]}.\n'
        f"Items:\n{json.dumps(items, ensure_ascii=False)}\n"
    )

    usage_total: dict[str, int] = {}
    last_err: Exception | None = None
    max_out = int(max_output_tokens)
    max_out_cap = max(max_out, 8192)
    max_out_bumps = 0
    schema = copy.deepcopy(AugmentedRows.model_json_schema())
    if _model_requires_property_ordering(model):
        _add_property_ordering(schema)

    for attempt in range(retries + 1):
        try:
            prompt_attempt = prompt
            if attempt > 0:
                prompt_attempt = (
                    prompt
                    + "\nIMPORTANT: Your previous response was invalid. Return STRICT valid JSON only. "
                    "No markdown, no extra text, escape newlines/quotes properly.\n"
                )
            def do_call(max_tokens: int) -> Any:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_attempt)])],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        max_output_tokens=max_tokens,
                    ),
                )
                _add_token_usage(usage_total, _extract_gemini_usage(resp))
                return resp

            resp = do_call(max_out)
            while _finish_reason_is_max_tokens(resp) and max_out_bumps < 2 and max_out < max_out_cap:
                max_out_bumps += 1
                max_out = min(max_out * 2, max_out_cap)
                resp = do_call(max_out)
            if _finish_reason_is_max_tokens(resp):
                raise ValueError("Gemini response truncated (finish_reason=MAX_TOKENS)")

            raw_text = (resp.text or "").strip()
            if not raw_text:
                raise ValueError("Empty Gemini response")
            parsed = AugmentedRows.model_validate_json(raw_text)
            rows = parsed.rows
            out = [r.model_dump() for r in rows]

            expected_ids = {str(it.get("id")) for it in items if isinstance(it, dict) and str(it.get("id") or "")}
            got_ids = {r.get("id") for r in out if isinstance(r, dict)}
            if expected_ids and got_ids != expected_ids:
                missing = sorted(expected_ids - got_ids)
                extra = sorted(got_ids - expected_ids)
                raise ValueError(f"Gemini output ids mismatch (missing={missing}, extra={extra})")
            return out, usage_total
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Gemini augmentation failed for batch of {len(items)} items") from last_err


def _gemini_augment_batch_resilient(
    *,
    client: Any,
    model: str,
    items: list[dict[str, Any]],
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not items:
        return [], {}

    def fallback_keywords(item: dict[str, Any]) -> list[str]:
        parts: list[str] = []
        desc = str(item.get("description") or "")
        if desc:
            parts.append(desc)
        ctx = item.get("context_descriptions")
        if isinstance(ctx, list):
            parts.extend(str(x) for x in ctx if str(x).strip())
        text = _norm_space(" ".join(parts))
        if not text:
            return []
        stop = {
            "and",
            "or",
            "of",
            "the",
            "a",
            "an",
            "for",
            "with",
            "without",
            "whether",
            "not",
            "other",
        }
        tokens: list[str] = []
        seen: set[str] = set()
        for raw in text.split():
            tok = raw.strip(".,;:()[]{}'\"").strip()
            if len(tok) < 3:
                continue
            key = tok.casefold()
            if key in stop or key in seen:
                continue
            seen.add(key)
            tokens.append(tok)
            if len(tokens) >= 8:
                break
        return tokens

    try:
        return _gemini_augment_batch(
            client=client,
            model=model,
            items=items,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
    except Exception:  # noqa: BLE001
        if len(items) == 1:
            item = items[0]
            fallback = {
                "id": item.get("id", ""),
                "augmented_description": "",
                "keywords": fallback_keywords(item),
            }
            try:
                single_out, usage = _gemini_augment_batch(
                    client=client,
                    model=model,
                    items=[item],
                    max_output_tokens=max_output_tokens,
                    retries=max(0, retries),
                )
                return (single_out or [fallback]), usage
            except Exception:  # noqa: BLE001
                return [fallback], {}

        _eprint(f"Gemini batch failed for {len(items)} items; splitting.")
        mid = max(1, len(items) // 2)
        left, usage_left = _gemini_augment_batch_resilient(
            client=client,
            model=model,
            items=items[:mid],
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        right, usage_right = _gemini_augment_batch_resilient(
            client=client,
            model=model,
            items=items[mid:],
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        usage_total: dict[str, int] = {}
        _add_token_usage(usage_total, usage_left)
        _add_token_usage(usage_total, usage_right)
        return left + right, usage_total


def _gemini_augment_attrs_batch(
    *,
    client: Any,
    model: str,
    items: list[dict[str, Any]],
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from google.genai import types
    from pydantic import BaseModel, Field

    class AugmentedAttrRow(BaseModel):
        id: str = Field(..., description="Stable input id for joining results back to the input.")
        keywords: list[str] = Field(
            default_factory=list,
            description="Search keywords/synonyms for the attribute triple (max 12).",
        )

    class AugmentedAttrRows(BaseModel):
        rows: list[AugmentedAttrRow]

    prompt = (
        "You improve multilingual search recall for product attribute triples.\n"
        "For each item below, generate multilingual keywords/synonyms to help matching user queries.\n"
        "Rules:\n"
        "- Do NOT invent products/attributes. Only use what is implied by the triple.\n"
        "- Provide 6-12 short keywords/phrases total.\n"
        "- Always include at least 4 English keywords.\n"
        f"- Target non-English languages: {_KEYWORD_LANGS_TARGET}.\n"
        "  - Use native scripts (Devanagari for Hindi; Arabic script for Urdu/Arabic; Cyrillic for Russian).\n"
        "  - Try to include at least 1 keyword in each target non-English language when you are confident.\n"
        "  - If unsure about a translation/transliteration, omit it (do not guess).\n"
        "  - Do not include other languages/scripts.\n"
        "- Include the original product name, attribute type, and attribute value as keywords.\n"
        "- Also include translations/synonyms for the attribute value when it is a common concept\n"
        "  (e.g., colors, organic/certified, grades, sizes/mesh, processing methods, packaging types, countries).\n"
        "  - If the value is numeric/code-like (e.g., '84.5', '28 Mesh', '570 G/L'), keep it as-is; optionally add a\n"
        "    short unit/meaning keyword in English.\n"
        "- For attribute types, include common variants (e.g. 'certification', 'certificate').\n"
        "- Avoid boilerplate/low-signal words.\n"
        'Return ONLY JSON object: {"rows":[{id, keywords}, ...]}.\n'
        f"Items:\n{json.dumps(items, ensure_ascii=False)}\n"
    )

    usage_total: dict[str, int] = {}
    last_err: Exception | None = None
    max_out = int(max_output_tokens)
    max_out_cap = max(max_out, 8192)
    max_out_bumps = 0
    schema = copy.deepcopy(AugmentedAttrRows.model_json_schema())
    if _model_requires_property_ordering(model):
        _add_property_ordering(schema)

    for attempt in range(retries + 1):
        try:
            prompt_attempt = prompt
            if attempt > 0:
                prompt_attempt = (
                    prompt
                    + "\nIMPORTANT: Your previous response was invalid. Return STRICT valid JSON only. "
                    "No markdown, no extra text, escape newlines/quotes properly.\n"
                )
            def do_call(max_tokens: int) -> Any:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_attempt)])],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        max_output_tokens=max_tokens,
                    ),
                )
                _add_token_usage(usage_total, _extract_gemini_usage(resp))
                return resp

            resp = do_call(max_out)
            while _finish_reason_is_max_tokens(resp) and max_out_bumps < 2 and max_out < max_out_cap:
                max_out_bumps += 1
                max_out = min(max_out * 2, max_out_cap)
                resp = do_call(max_out)
            if _finish_reason_is_max_tokens(resp):
                raise ValueError("Gemini response truncated (finish_reason=MAX_TOKENS)")

            raw_text = (resp.text or "").strip()
            if not raw_text:
                raise ValueError("Empty Gemini response")
            parsed = AugmentedAttrRows.model_validate_json(raw_text)
            rows = parsed.rows
            out = [r.model_dump() for r in rows]

            expected_ids = {str(it.get("id")) for it in items if isinstance(it, dict) and str(it.get("id") or "")}
            got_ids = {r.get("id") for r in out if isinstance(r, dict)}
            if expected_ids and got_ids != expected_ids:
                missing = sorted(expected_ids - got_ids)
                extra = sorted(got_ids - expected_ids)
                raise ValueError(f"Gemini output ids mismatch (missing={missing}, extra={extra})")
            return out, usage_total
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Gemini augmentation failed for batch of {len(items)} items") from last_err


def _gemini_augment_terms_batch(
    *,
    client: Any,
    model: str,
    kind: str,
    items: list[dict[str, Any]],
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from google.genai import types
    from pydantic import BaseModel, Field

    class AugmentedTermRow(BaseModel):
        id: str = Field(..., description="Stable input id for joining results back to the input.")
        keywords: list[str] = Field(default_factory=list, description="Search keywords/synonyms (max 12).")

    class AugmentedTermRows(BaseModel):
        rows: list[AugmentedTermRow]

    if kind not in {"product", "attribute_type"}:
        raise ValueError(f"Unknown terms kind: {kind}")

    kind_label = "product name" if kind == "product" else "attribute type"
    example_block = ""
    if kind == "product":
        example_block = (
            "Examples:\n"
            "- Black Pepper -> black pepper; peppercorn; Piper nigrum; kali mirch; काली मिर्च; کالی مرچ; فلفل أسود; "
            "poivre noir; pimienta negra; чёрный перец; tiêu đen\n"
            "- Red Chilli -> red chili; chili pepper; Capsicum; lal mirch; लाल मिर्च; لال مرچ; فلفل حار أحمر; "
            "piment rouge; chile rojo; перец чили; ớt đỏ\n"
        )
    else:
        example_block = (
            "Examples:\n"
            "- Certifications -> certifications; certification; certificate; certified; प्रमाणन; سرٹیفکیشن; شهادة; "
            "certification; certificación; сертификация; chứng nhận\n"
            "- Variety -> variety; type; kind; किस्म; قسم; نوع; variété; variedad; разновидность; giống\n"
        )

    prompt = (
        "You generate multilingual search keywords/synonyms to improve retrieval.\n"
        f"For each {kind_label} below, output short keywords/synonyms.\n"
        "Rules:\n"
        "- Do NOT invent unrelated terms.\n"
        "- Provide 6-12 keywords/phrases total.\n"
        "- Always include at least 4 English keywords.\n"
        f"- Target non-English languages: {_KEYWORD_LANGS_TARGET}.\n"
        "  - Use native scripts (Devanagari for Hindi; Arabic script for Urdu/Arabic; Cyrillic for Russian).\n"
        "  - Try to include at least 1 keyword in each target non-English language when you are confident.\n"
        "  - If unsure about a translation/transliteration, omit it (do not guess).\n"
        "  - Do not include other languages/scripts.\n"
        "- Always include the original term verbatim as a keyword.\n"
        "- Avoid boilerplate/low-signal words.\n"
        f"{example_block}"
        'Return ONLY JSON object: {"rows":[{id, keywords}, ...]}.\n'
        f"Items:\n{json.dumps(items, ensure_ascii=False)}\n"
    )

    usage_total: dict[str, int] = {}
    last_err: Exception | None = None
    max_out = int(max_output_tokens)
    max_out_cap = max(max_out, 8192)
    max_out_bumps = 0
    schema = copy.deepcopy(AugmentedTermRows.model_json_schema())
    if _model_requires_property_ordering(model):
        _add_property_ordering(schema)

    for attempt in range(retries + 1):
        try:
            prompt_attempt = prompt
            if attempt > 0:
                prompt_attempt = (
                    prompt
                    + "\nIMPORTANT: Your previous response was invalid. Return STRICT valid JSON only. "
                    "No markdown, no extra text, escape newlines/quotes properly.\n"
                )
            def do_call(max_tokens: int) -> Any:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_attempt)])],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        max_output_tokens=max_tokens,
                    ),
                )
                _add_token_usage(usage_total, _extract_gemini_usage(resp))
                return resp

            resp = do_call(max_out)
            while _finish_reason_is_max_tokens(resp) and max_out_bumps < 2 and max_out < max_out_cap:
                max_out_bumps += 1
                max_out = min(max_out * 2, max_out_cap)
                resp = do_call(max_out)
            if _finish_reason_is_max_tokens(resp):
                raise ValueError("Gemini response truncated (finish_reason=MAX_TOKENS)")

            raw_text = (resp.text or "").strip()
            if not raw_text:
                raise ValueError("Empty Gemini response")
            parsed = AugmentedTermRows.model_validate_json(raw_text)
            rows = parsed.rows
            out = [r.model_dump() for r in rows]

            expected_ids = {str(it.get("id")) for it in items if isinstance(it, dict) and str(it.get("id") or "")}
            got_ids = {r.get("id") for r in out if isinstance(r, dict)}
            if expected_ids and got_ids != expected_ids:
                missing = sorted(expected_ids - got_ids)
                extra = sorted(got_ids - expected_ids)
                raise ValueError(f"Gemini output ids mismatch (missing={missing}, extra={extra})")
            return out, usage_total
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Gemini term augmentation failed for batch of {len(items)} items") from last_err


def _gemini_augment_terms_batch_resilient(
    *,
    client: Any,
    model: str,
    kind: str,
    items: list[dict[str, Any]],
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not items:
        return [], {}

    def fallback_keywords(item: dict[str, Any]) -> list[str]:
        term = _norm_space(str(item.get("term") or item.get("id") or ""))
        if not term:
            return []
        return [term]

    try:
        return _gemini_augment_terms_batch(
            client=client,
            model=model,
            kind=kind,
            items=items,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
    except Exception:  # noqa: BLE001
        if len(items) == 1:
            item = items[0]
            fallback = {"id": item.get("id", ""), "keywords": fallback_keywords(item)}
            try:
                single_out, usage = _gemini_augment_terms_batch(
                    client=client,
                    model=model,
                    kind=kind,
                    items=[item],
                    max_output_tokens=max_output_tokens,
                    retries=max(0, retries),
                )
                return (single_out or [fallback]), usage
            except Exception:  # noqa: BLE001
                return [fallback], {}

        _eprint(f"Gemini term batch failed for {len(items)} items; splitting.")
        mid = max(1, len(items) // 2)
        left, usage_left = _gemini_augment_terms_batch_resilient(
            client=client,
            model=model,
            kind=kind,
            items=items[:mid],
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        right, usage_right = _gemini_augment_terms_batch_resilient(
            client=client,
            model=model,
            kind=kind,
            items=items[mid:],
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        usage_total: dict[str, int] = {}
        _add_token_usage(usage_total, usage_left)
        _add_token_usage(usage_total, usage_right)
        return left + right, usage_total


def _augment_terms_with_llm(
    *,
    terms: list[str],
    kind: str,
    model: str,
    batch_size: int,
    workers: int,
    cache_path: Path,
    max_output_tokens: int,
    retries: int,
    throttle_seconds: float,
    force: bool,
) -> dict[str, list[str]]:
    cache = _load_jsonl_cache(cache_path)
    pending: list[dict[str, Any]] = []
    skipped_cached = 0

    uniq: list[str] = []
    seen: set[str] = set()
    for t in terms:
        t_clean = _norm_space(str(t or ""))
        if not t_clean or t_clean in seen:
            continue
        seen.add(t_clean)
        uniq.append(t_clean)

    for t in uniq:
        if not force:
            cached = cache.get(t)
            cached_keywords = cached.get("keywords") if isinstance(cached, dict) else None
            if isinstance(cached_keywords, list) and len(cached_keywords) > 0:
                skipped_cached += 1
                continue
        pending.append({"id": t, "term": t})

    from google import genai

    token_totals: dict[str, int] = {}
    if pending:
        api_key = _load_google_api_key()
        max_workers = max(1, int(workers))
        _eprint(
            f"{kind}: augmenting {len(pending)} terms with LLM ({model}) "
            f"(cached={skipped_cached}, workers={max_workers}, batch={batch_size})"
        )
        pbar = tqdm(total=len(pending), desc=f"{kind}: LLM augment", unit="term", file=sys.stderr)
        batches = _chunks(pending, batch_size)
        thread_local = threading.local()

        def get_client() -> Any:
            client = getattr(thread_local, "client", None)
            if client is None:
                client = genai.Client(api_key=api_key)
                thread_local.client = client
            return client

        def run_one(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
            out, usage = _gemini_augment_terms_batch_resilient(
                client=get_client(),
                model=model,
                kind=kind,
                items=batch,
                max_output_tokens=max_output_tokens,
                retries=retries,
            )
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            return out, usage

        def merge_rows(rows_out: list[dict[str, Any]]) -> None:
            updated = False
            for row in rows_out:
                rid = row.get("id")
                if not isinstance(rid, str) or not rid:
                    continue
                kw = row.get("keywords") if isinstance(row.get("keywords"), list) else []
                merged = {"id": rid, "model": model, "keywords": _dedupe_keywords(kw, limit=12)}
                cache[rid] = merged
                updated = True
            if updated:
                _write_jsonl_cache_latest(cache_path, cache)

        if max_workers <= 1 or len(batches) <= 1:
            for batch in batches:
                rows_out, usage = run_one(batch)
                _add_token_usage(token_totals, usage)
                pbar.update(len(batch))
                merge_rows(rows_out)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_to_batch_len = {ex.submit(run_one, batch): len(batch) for batch in batches}
                for fut in as_completed(future_to_batch_len):
                    batch_len = future_to_batch_len[fut]
                    try:
                        rows_out, usage = fut.result()
                    except Exception as e:  # noqa: BLE001
                        _eprint(f"{kind}: LLM batch failed: {e}")
                        pbar.update(batch_len)
                        continue
                    _add_token_usage(token_totals, usage)
                    pbar.update(batch_len)
                    merge_rows(rows_out)

        pbar.close()
        if token_totals:
            _eprint(
                f"{kind}: token totals prompt={token_totals.get('prompt_tokens', 0)} "
                f"completion={token_totals.get('completion_tokens', 0)} total={token_totals.get('total_tokens', 0)}"
            )
    elif cache_path.exists():
        # Normalize legacy append-only caches to "latest per id".
        _write_jsonl_cache_latest(cache_path, cache)

    out: dict[str, list[str]] = {}
    for t in uniq:
        cached = cache.get(t)
        kw = cached.get("keywords") if isinstance(cached, dict) and isinstance(cached.get("keywords"), list) else []
        out[t] = _dedupe_keywords(list(kw), limit=12)
    return out


def _gemini_augment_attrs_batch_resilient(
    *,
    client: Any,
    model: str,
    items: list[dict[str, Any]],
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not items:
        return [], {}

    def fallback_keywords(item: dict[str, Any]) -> list[str]:
        parts = [
            str(item.get("product") or ""),
            str(item.get("attribute_type") or ""),
            str(item.get("attribute_value") or ""),
            str(item.get("text") or ""),
        ]
        text = _norm_space(" ".join([p for p in parts if p.strip()]))
        if not text:
            return []
        toks: list[str] = []
        seen: set[str] = set()
        for raw in text.replace(":", " ").split():
            tok = raw.strip(".,;:()[]{}'\"").strip()
            if len(tok) < 3:
                continue
            key = tok.casefold()
            if key in seen:
                continue
            seen.add(key)
            toks.append(tok)
            if len(toks) >= 10:
                break
        return toks

    try:
        return _gemini_augment_attrs_batch(
            client=client,
            model=model,
            items=items,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
    except Exception:  # noqa: BLE001
        if len(items) == 1:
            item = items[0]
            fallback = {"id": item.get("id", ""), "keywords": fallback_keywords(item)}
            try:
                single_out, usage = _gemini_augment_attrs_batch(
                    client=client,
                    model=model,
                    items=[item],
                    max_output_tokens=max_output_tokens,
                    retries=max(0, retries),
                )
                return (single_out or [fallback]), usage
            except Exception:  # noqa: BLE001
                return [fallback], {}

        _eprint(f"Gemini attr batch failed for {len(items)} items; splitting.")
        mid = max(1, len(items) // 2)
        left, usage_left = _gemini_augment_attrs_batch_resilient(
            client=client,
            model=model,
            items=items[:mid],
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        right, usage_right = _gemini_augment_attrs_batch_resilient(
            client=client,
            model=model,
            items=items[mid:],
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        usage_total: dict[str, int] = {}
        _add_token_usage(usage_total, usage_left)
        _add_token_usage(usage_total, usage_right)
        return left + right, usage_total


def _augment_attributes_with_llm(
    *,
    objects: list[dict[str, Any]],
    model: str,
    batch_size: int,
    workers: int,
    product_terms: dict[str, list[str]] | None,
    attribute_type_terms: dict[str, list[str]] | None,
    cache_path: Path,
    max_output_tokens: int,
    retries: int,
    throttle_seconds: float,
    force: bool,
) -> None:
    if not objects:
        return

    cache = _load_jsonl_cache(cache_path)
    if cache:
        # Normalize legacy cache ids that used other delimiters to the current delimiter (\t).
        def _norm_cache_id(cache_id: str) -> str:
            return (cache_id or "").replace("\u0000", "\t").replace("\n", "\t").replace("\r", "")

        for k in list(cache.keys()):
            if not isinstance(k, str) or not k:
                continue
            nk = _norm_cache_id(k)
            if not nk or nk == k:
                continue
            current = cache.get(nk)
            cand = cache.get(k)
            if not isinstance(cand, dict):
                cache.pop(k, None)
                continue

            def _quality(v: Any) -> int:
                if not isinstance(v, dict):
                    return 0
                kw = v.get("keywords")
                return len(kw) if isinstance(kw, list) else 0

            if current is None or _quality(cand) >= _quality(current):
                cache[nk] = cand
            cache.pop(k, None)

    def obj_id(obj: dict[str, Any]) -> str:
        # Keep this stable for cache lookups; attrs.json fields are plain strings (no tabs).
        return f"{obj.get('product','')}\t{obj.get('attribute_type','')}\t{obj.get('attribute_value','')}"

    def obj_id_legacy(obj: dict[str, Any]) -> str:
        # Back-compat with an older cache delimiter.
        return f"{obj.get('product','')}\u0000{obj.get('attribute_type','')}\u0000{obj.get('attribute_value','')}"

    pending: list[dict[str, Any]] = []
    skipped_cached = 0
    for obj in objects:
        rid = obj_id(obj)
        rid_legacy = obj_id_legacy(obj)
        if not rid.strip():
            continue
        if not force:
            cached = cache.get(rid) or cache.get(rid_legacy)
            cached_keywords = cached.get("keywords") if isinstance(cached, dict) else None
            if isinstance(cached_keywords, list) and len(cached_keywords) > 0:
                skipped_cached += 1
                continue
        pending.append(
            {
                "id": rid,
                "product": obj.get("product"),
                "attribute_type": obj.get("attribute_type"),
                "attribute_value": obj.get("attribute_value"),
                "text": obj.get("text"),
            }
        )

    from google import genai

    token_totals: dict[str, int] = {}
    if pending:
        api_key = _load_google_api_key()
        _eprint(
            f"Attributes: augmenting {len(pending)} triples with LLM ({model}) "
            f"(cached={skipped_cached}, workers={max(1, int(workers))}, batch={batch_size})"
        )
        pbar = tqdm(total=len(pending), desc="Attributes: LLM augment", unit="row", file=sys.stderr)
        batches = _chunks(pending, batch_size)
        max_workers = max(1, int(workers))

        thread_local = threading.local()

        def get_client() -> Any:
            client = getattr(thread_local, "client", None)
            if client is None:
                client = genai.Client(api_key=api_key)
                thread_local.client = client
            return client

        def run_one(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
            out, usage = _gemini_augment_attrs_batch_resilient(
                client=get_client(),
                model=model,
                items=batch,
                max_output_tokens=max_output_tokens,
                retries=retries,
            )
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            return out, usage

        def merge_rows(rows_out: list[dict[str, Any]]) -> None:
            updated = False
            for row in rows_out:
                rid = row.get("id")
                if not isinstance(rid, str) or not rid:
                    continue
                existing = cache.get(rid) if isinstance(cache.get(rid), dict) else {}
                merged: dict[str, Any] = {"id": rid, "model": model}
                kw = row.get("keywords")
                if isinstance(kw, list) and (
                    kw or not (isinstance(existing, dict) and isinstance(existing.get("keywords"), list))
                ):
                    merged["keywords"] = _dedupe_keywords(kw, limit=12)
                elif isinstance(existing, dict) and "keywords" in existing:
                    merged["keywords"] = _dedupe_keywords(list(existing.get("keywords", [])), limit=12)

                cache[rid] = merged
                updated = True

            if updated:
                _write_jsonl_cache_latest(cache_path, cache)

        if max_workers <= 1 or len(batches) <= 1:
            for batch in batches:
                rows_out, usage = run_one(batch)
                _add_token_usage(token_totals, usage)
                pbar.update(len(batch))
                merge_rows(rows_out)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_to_batch_len = {ex.submit(run_one, batch): len(batch) for batch in batches}
                for fut in as_completed(future_to_batch_len):
                    batch_len = future_to_batch_len[fut]
                    try:
                        rows_out, usage = fut.result()
                    except Exception as e:  # noqa: BLE001
                        _eprint(f"Attributes: LLM batch failed: {e}")
                        pbar.update(batch_len)
                        continue
                    _add_token_usage(token_totals, usage)
                    pbar.update(batch_len)
                    merge_rows(rows_out)

        pbar.close()
        _eprint(
            f"Attributes: token totals prompt={token_totals.get('prompt_tokens', 0)} "
            f"completion={token_totals.get('completion_tokens', 0)} total={token_totals.get('total_tokens', 0)}"
        )
    elif skipped_cached:
        _eprint(f"Attributes: using cached keywords for {skipped_cached} triples (no LLM calls)")
        if cache_path.exists():
            _write_jsonl_cache_latest(cache_path, cache)

    def build_search_text(obj: dict[str, Any], *, keywords: list[str]) -> str:
        base = _norm_space(str(obj.get("text") or ""))
        if not keywords:
            return base
        kw = ", ".join(keywords)
        return _norm_space(f"{base} Keywords: {kw}")

    for obj in objects:
        obj.setdefault("keywords", [])
        rid = obj_id(obj)
        rid_legacy = obj_id_legacy(obj)
        aug = cache.get(rid) or cache.get(rid_legacy)
        if not isinstance(aug, dict):
            base_kw = list(obj.get("keywords") or [])
            prod = _norm_space(str(obj.get("product") or ""))
            atype = _norm_space(str(obj.get("attribute_type") or ""))
            merged: list[str] = []
            if prod and isinstance(product_terms, dict):
                merged.extend(product_terms.get(prod) or [])
            if atype and isinstance(attribute_type_terms, dict):
                merged.extend(attribute_type_terms.get(atype) or [])
            merged.extend(base_kw)
            merged_clean = [_norm_space(str(k)) for k in merged if _norm_space(str(k))]
            deduped: list[str] = []
            seen: set[str] = set()
            for k in merged_clean:
                key = k.casefold()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(k)
            final_kw = deduped[:20]
            obj["keywords"] = final_kw
            obj["text"] = build_search_text(obj, keywords=final_kw)
            continue
        keywords = aug.get("keywords") if isinstance(aug.get("keywords"), list) else []
        keywords_clean = _dedupe_keywords(list(keywords), limit=20)
        prod = _norm_space(str(obj.get("product") or ""))
        atype = _norm_space(str(obj.get("attribute_type") or ""))
        merged_in: list[str] = []
        if prod and isinstance(product_terms, dict):
            merged_in.extend(product_terms.get(prod) or [])
        if atype and isinstance(attribute_type_terms, dict):
            merged_in.extend(attribute_type_terms.get(atype) or [])
        merged_in.extend(keywords_clean)
        keywords_clean = _dedupe_keywords(merged_in, limit=20)
        obj["keywords"] = keywords_clean
        obj["text"] = build_search_text(obj, keywords=keywords_clean)


def _augment_objects_with_llm(
    *,
    objects: list[dict[str, Any]],
    kind: str,
    model: str,
    mode: str,
    threshold: int,
    batch_size: int,
    workers: int,
    cache_path: Path,
    desc_by_digits: dict[str, str] | None,
    max_output_tokens: int,
    retries: int,
    throttle_seconds: float,
    force: bool,
) -> None:
    if not objects:
        return

    cache = _load_jsonl_cache(cache_path)

    def obj_id(obj: dict[str, Any]) -> str:
        if kind == "HS6":
            return str(obj.get("digits") or "")
        if kind == "HSCountry":
            return f"{obj.get('country','')}|{obj.get('digits','')}"
        raise ValueError(f"Unknown kind: {kind}")

    def context_descriptions(obj: dict[str, Any]) -> list[str]:
        if not desc_by_digits:
            return []
        digits = str(obj.get("digits") or "")
        if len(digits) < 4:
            return []
        ctx: list[str] = []
        for n in (8, 6, 4, 2):
            if len(digits) <= n:
                continue
            d = desc_by_digits.get(digits[:n])
            if not d:
                continue
            d_clean = _norm_space(d)
            if not d_clean or d_clean in ctx:
                continue
            ctx.append(d_clean)
            if len(ctx) >= 2:
                break
        return ctx

    pending: list[dict[str, Any]] = []
    skipped_cached = 0
    skipped_no_rewrite = 0
    for obj in objects:
        current_desc = str(obj.get("description") or "")
        o_id = obj_id(obj)
        if not o_id:
            continue
        need_rewrite = mode == "all" or _needs_description_rewrite(current_desc, threshold=threshold)
        need_keywords = mode in {"all", "keywords"} or (mode == "short" and need_rewrite)
        if mode == "short" and not need_rewrite:
            skipped_no_rewrite += 1
            continue

        if not force:
            cached = cache.get(o_id)
            if cached is not None:
                cached_aug = str(cached.get("augmented_description") or "")
                has_aug = bool(cached_aug.strip())
                cached_keywords = cached.get("keywords")
                has_keywords = isinstance(cached_keywords, list) and len(cached_keywords) > 0
                if (not need_rewrite or has_aug) and (not need_keywords or has_keywords):
                    skipped_cached += 1
                    continue

        pending.append(
            {
                "id": o_id,
                "hs_code": obj.get("hs_code"),
                "country": obj.get("country") if kind == "HSCountry" else "",
                "description": current_desc,
                "description_segments": _split_description_segments(current_desc),
                "context_descriptions": context_descriptions(obj),
                "rewrite_description": need_rewrite,
            }
        )

    from google import genai
    token_totals: dict[str, int] = {}
    if pending:
        api_key = _load_google_api_key()
        max_workers = max(1, int(workers))
        _eprint(
            f"{kind}: augmenting {len(pending)} descriptions with LLM ({model}) "
            f"(cached={skipped_cached}, skipped_no_rewrite={skipped_no_rewrite}, workers={max_workers}, batch={batch_size})"
        )

        pbar = tqdm(total=len(pending), desc=f"{kind}: LLM augment", unit="row", file=sys.stderr)
        batches = _chunks(pending, batch_size)
        thread_local = threading.local()

        def get_client() -> Any:
            client = getattr(thread_local, "client", None)
            if client is None:
                client = genai.Client(api_key=api_key)
                thread_local.client = client
            return client

        def run_one(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
            out, usage = _gemini_augment_batch_resilient(
                client=get_client(),
                model=model,
                items=batch,
                max_output_tokens=max_output_tokens,
                retries=retries,
            )
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            return out, usage

        def merge_rows(rows_out: list[dict[str, Any]]) -> None:
            updated = False
            for row in rows_out:
                rid = row.get("id")
                if not isinstance(rid, str) or not rid:
                    continue
                existing = cache.get(rid) if isinstance(cache.get(rid), dict) else {}
                merged: dict[str, Any] = {"id": rid, "model": model}

                aug_desc = row.get("augmented_description")
                if isinstance(aug_desc, str) and aug_desc.strip():
                    merged["augmented_description"] = _norm_space(aug_desc)
                elif isinstance(existing, dict):
                    prev_aug = existing.get("augmented_description")
                    if isinstance(prev_aug, str) and prev_aug.strip():
                        merged["augmented_description"] = prev_aug

                kw = row.get("keywords")
                if isinstance(kw, list) and (
                    kw or not (isinstance(existing, dict) and isinstance(existing.get("keywords"), list))
                ):
                    merged["keywords"] = _dedupe_keywords(kw, limit=12)
                elif isinstance(existing, dict) and "keywords" in existing:
                    merged["keywords"] = _dedupe_keywords(list(existing.get("keywords", [])), limit=12)

                cache[rid] = merged
                updated = True
            if updated:
                _write_jsonl_cache_latest(cache_path, cache)

        if max_workers <= 1 or len(batches) <= 1:
            for batch in batches:
                rows_out, usage = run_one(batch)
                _add_token_usage(token_totals, usage)
                pbar.update(len(batch))
                merge_rows(rows_out)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_to_batch_len = {ex.submit(run_one, batch): len(batch) for batch in batches}
                for fut in as_completed(future_to_batch_len):
                    batch_len = future_to_batch_len[fut]
                    try:
                        rows_out, usage = fut.result()
                    except Exception as e:  # noqa: BLE001
                        _eprint(f"{kind}: LLM batch failed: {e}")
                        pbar.update(batch_len)
                        continue
                    _add_token_usage(token_totals, usage)
                    pbar.update(batch_len)
                    merge_rows(rows_out)

        pbar.close()

        if token_totals:
            _eprint(
                f"{kind}: token totals prompt={token_totals.get('prompt_tokens', 0)} "
                f"completion={token_totals.get('completion_tokens', 0)} total={token_totals.get('total_tokens', 0)}"
            )
    elif cache_path.exists():
        # Normalize legacy append-only caches to "latest per id".
        _write_jsonl_cache_latest(cache_path, cache)

    # Apply cache to objects (also sets description_raw/keywords consistently)
    def build_search_text(
        obj: dict[str, Any],
        *,
        description: str,
        description_raw: str,
        keywords: list[str],
    ) -> str:
        parts: list[str] = []
        if kind == "HSCountry":
            country = _norm_space(str(obj.get("country") or ""))
            if country:
                parts.append(country)
        hs_code = _norm_space(str(obj.get("hs_code") or ""))
        if hs_code:
            parts.append(hs_code)
        digits = _norm_space(str(obj.get("digits") or ""))
        if digits:
            parts.append(digits)
        desc_clean = _norm_space(description)
        if desc_clean:
            parts.append(desc_clean)
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")
        return _norm_space(" ".join(parts))

    for obj in objects:
        raw = str(obj.get("description") or "")
        obj.setdefault("description_raw", raw)
        obj.setdefault("keywords", [])
        o_id = obj_id(obj)
        aug = cache.get(o_id)
        if not aug:
            obj["text"] = build_search_text(
                obj,
                description=raw,
                description_raw=str(obj.get("description_raw") or raw),
                keywords=list(obj.get("keywords") or []),
            )
            continue

        aug_desc = _norm_space(str(aug.get("augmented_description") or raw))
        keywords = aug.get("keywords") if isinstance(aug.get("keywords"), list) else []
        keywords_clean = _dedupe_keywords(list(keywords), limit=12)

        obj["description"] = aug_desc
        obj["keywords"] = keywords_clean

        obj["text"] = build_search_text(
            obj,
            description=aug_desc,
            description_raw=str(obj.get("description_raw") or raw),
            keywords=keywords_clean,
        )


def build_hs6_objects(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    best_by_digits: dict[str, dict[str, Any]] = {}
    for r in rows:
        hs_code = (r.get("hs_code") or "").strip()
        description = (r.get("description") or "").strip()
        pdf = (r.get("pdf") or "").strip()
        page_raw = (r.get("page") or "").strip()
        digits = digits_only(hs_code)
        if len(digits) != 6:
            continue

        try:
            page = int(page_raw)
        except Exception:  # noqa: BLE001
            page = 0

        key = digits
        prev = best_by_digits.get(key)
        if not prev or len(description) > len(str(prev.get("description", ""))):
            best_by_digits[key] = {
                "text": f"{hs_code} {digits} {description}".strip(),
                "hs_code": hs_code,
                "description": description,
                "description_raw": description,
                "keywords": [],
                "digits": digits,
                "source_pdf": pdf,
                "page": page,
            }
    return list(best_by_digits.values())


def build_hs_country_objects(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for r in rows:
        hs_code = (r.get("hs_code") or "").strip()
        description = (r.get("description") or "").strip()
        pdf = (r.get("pdf") or "").strip()
        page_raw = (r.get("page") or "").strip()
        digits = digits_only(hs_code)
        if not (8 <= len(digits) <= 12):
            continue
        if not is_country_specific_pdf(pdf):
            continue

        country = (r.get("country") or "").strip() or infer_country_from_pdf_path(pdf)
        try:
            page = int(page_raw)
        except Exception:  # noqa: BLE001
            page = 0

        key = f"{country}|{digits}"
        prev = best_by_key.get(key)
        if not prev or len(description) > len(str(prev.get("description", ""))):
            best_by_key[key] = {
                "text": f"{country} {hs_code} {digits} {description}".strip(),
                "country": country,
                "hs_code": hs_code,
                "description": description,
                "description_raw": description,
                "keywords": [],
                "digits": digits,
                "source_pdf": pdf,
                "page": page,
            }
    return list(best_by_key.values())


def load_attribute_triples(attrs_json_path: Path) -> list[dict[str, Any]]:
    if not attrs_json_path.exists():
        raise FileNotFoundError(f"Missing attrs JSON: {attrs_json_path}")
    obj = json.loads(attrs_json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("attrs.json must be a JSON object mapping product -> schema")

    seen: set[str] = set()
    triples: list[dict[str, Any]] = []
    for product, meta in obj.items():
        if not isinstance(product, str) or not isinstance(meta, dict):
            continue
        schema_id = str(meta.get("schema_id") or "")
        source_file = str(meta.get("file") or "")
        schema = meta.get("schema")
        if not isinstance(schema, dict):
            continue
        for attribute_type, values in schema.items():
            if not isinstance(attribute_type, str) or not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str):
                    continue
                key = f"{product}\u0000{attribute_type}\u0000{value}"
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    {
                        "text": f"{product}: {attribute_type} : {value}",
                        "product": product,
                        "attribute_type": attribute_type,
                        "attribute_value": value,
                        "keywords": [],
                        "schema_id": schema_id,
                        "source_file": source_file,
                    }
                )
    return triples


def ensure_collections(client: Any, *, recreate: bool) -> None:
    specs: list[tuple[str, list[Property]]] = [
        (
            "HS6",
            [
                Property(name="text", data_type=DataType.TEXT),
                Property(name="hs_code", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="description", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="description_raw", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
                Property(name="digits", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="source_pdf", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="page", data_type=DataType.INT, skip_vectorization=True),
            ],
        ),
        (
            "HSCountry",
            [
                Property(name="text", data_type=DataType.TEXT),
                Property(name="country", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="hs_code", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="description", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="description_raw", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
                Property(name="digits", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="source_pdf", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="page", data_type=DataType.INT, skip_vectorization=True),
            ],
        ),
        (
            "Attributes",
            [
                Property(name="text", data_type=DataType.TEXT),
                Property(name="product", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="attribute_type", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="attribute_value", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
                Property(name="schema_id", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="source_file", data_type=DataType.TEXT, skip_vectorization=True),
            ],
        ),
    ]

    for name, properties in specs:
        exists = client.collections.exists(name)
        if exists and recreate:
            _eprint(f"Deleting collection: {name}")
            client.collections.delete(name)
            exists = False
        if not exists:
            _eprint(f"Creating collection: {name}")
            client.collections.create(
                name=name,
                vector_config=Configure.Vectors.text2vec_transformers(),
                reranker_config=Configure.Reranker.transformers(),
                properties=properties,
            )


def batch_import(client: Any, *, name: str, objects: list[dict[str, Any]], batch_size: int) -> None:
    if not objects:
        _eprint(f"{name}: nothing to import")
        return

    collection = client.collections.get(name)
    _eprint(f"{name}: importing {len(objects)} objects")

    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for obj in tqdm(objects, desc=f"{name}: import", unit="obj", file=sys.stderr):
            if name == "HS6":
                seed = str(obj.get("digits") or obj.get("hs_code") or obj.get("text") or "")
            elif name == "HSCountry":
                seed = f"{obj.get('country','')}|{obj.get('digits','')}"
            else:
                seed = f"{obj.get('product','')}|{obj.get('attribute_type','')}|{obj.get('attribute_value','')}"
            uuid = generate_uuid5(seed)
            batch.add_object(properties=obj, uuid=uuid)

    failed = getattr(collection.batch, "failed_objects", None)
    if failed:
        _eprint(f"{name}: failed imports: {len(failed)} (showing up to 5)")
        for f in failed[:5]:
            _eprint(f"  - {getattr(f, 'message', f)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index HS codes + attributes into Weaviate.")
    parser.add_argument("--hs-csv", type=Path, default=Path("outputs/hs_rows_gemini.csv"))
    parser.add_argument("--attrs-json", type=Path, default=Path("attrs.json"))
    parser.add_argument("--weaviate-url", type=str, default=None)
    parser.add_argument("--weaviate-api-key", type=str, default=None)
    parser.add_argument("--grpc-port", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--llm-augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Gemini to augment HS descriptions for better search (default: False).",
    )
    parser.add_argument(
        "--llm-force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate LLM keywords/rewrites even if present in the cache (default: False).",
    )
    parser.add_argument(
        "--llm-augment-attrs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Gemini to add multilingual keywords to attrs.json triples before indexing (default: False).",
    )
    parser.add_argument("--llm-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument(
        "--llm-mode",
        type=str,
        choices=["short", "all", "keywords"],
        default="keywords",
        help=(
            "LLM augmentation mode: rewrite only short/generic descriptions ('short'), rewrite all ('all'), "
            "or generate multilingual keywords for all while rewriting only short/generic ('keywords', default)."
        ),
    )
    parser.add_argument(
        "--llm-threshold",
        type=int,
        default=80,
        help=(
            "Rewrite HS descriptions shorter than this threshold (also rewrites generic/empty ones) in "
            "--llm-mode short/keywords (default: 80)."
        ),
    )
    parser.add_argument("--llm-batch-size", type=int, default=20)
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=4,
        help="Number of parallel LLM requests to run (default: 4).",
    )
    parser.add_argument(
        "--llm-cache-jsonl",
        type=Path,
        default=Path(".cache/hs_augment/augmented_descriptions.jsonl"),
        help="JSONL cache of augmented descriptions (default: .cache/hs_augment/augmented_descriptions.jsonl).",
    )
    parser.add_argument(
        "--llm-attrs-cache-jsonl",
        type=Path,
        default=Path(".cache/attrs_augment/augmented_attributes.jsonl"),
        help="JSONL cache of augmented attribute keywords (default: .cache/attrs_augment/augmented_attributes.jsonl).",
    )
    parser.add_argument(
        "--llm-attrs-products-cache-jsonl",
        type=Path,
        default=Path(".cache/attrs_augment/augmented_products.jsonl"),
        help="JSONL cache of multilingual product synonyms (default: .cache/attrs_augment/augmented_products.jsonl).",
    )
    parser.add_argument(
        "--llm-attrs-types-cache-jsonl",
        type=Path,
        default=Path(".cache/attrs_augment/augmented_attribute_types.jsonl"),
        help=(
            "JSONL cache of multilingual attribute-type synonyms "
            "(default: .cache/attrs_augment/augmented_attribute_types.jsonl)."
        ),
    )
    parser.add_argument("--llm-max-output-tokens", type=int, default=2048)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--llm-throttle-seconds", type=float, default=0.2)
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop and recreate the collections before indexing (default: False).",
    )
    args = parser.parse_args()

    hs_rows = load_hs_rows(args.hs_csv)
    desc_by_digits = build_best_desc_by_digits(hs_rows)
    hs6_objs = build_hs6_objects(hs_rows)
    hs_country_objs = build_hs_country_objects(hs_rows)
    attr_objs = load_attribute_triples(args.attrs_json)

    if args.llm_augment:
        _augment_objects_with_llm(
            objects=hs6_objs,
            kind="HS6",
            model=args.llm_model,
            mode=args.llm_mode,
            threshold=args.llm_threshold,
            batch_size=args.llm_batch_size,
            workers=args.llm_workers,
            cache_path=args.llm_cache_jsonl,
            desc_by_digits=desc_by_digits,
            max_output_tokens=args.llm_max_output_tokens,
            retries=args.llm_retries,
            throttle_seconds=args.llm_throttle_seconds,
            force=args.llm_force,
        )
        _augment_objects_with_llm(
            objects=hs_country_objs,
            kind="HSCountry",
            model=args.llm_model,
            mode=args.llm_mode,
            threshold=args.llm_threshold,
            batch_size=args.llm_batch_size,
            workers=args.llm_workers,
            cache_path=args.llm_cache_jsonl,
            desc_by_digits=desc_by_digits,
            max_output_tokens=args.llm_max_output_tokens,
            retries=args.llm_retries,
            throttle_seconds=args.llm_throttle_seconds,
            force=args.llm_force,
        )
    if args.llm_augment_attrs:
        products = sorted({_norm_space(str(o.get("product") or "")) for o in attr_objs if _norm_space(str(o.get("product") or ""))})
        types = sorted(
            {_norm_space(str(o.get("attribute_type") or "")) for o in attr_objs if _norm_space(str(o.get("attribute_type") or ""))}
        )
        product_terms = _augment_terms_with_llm(
            terms=products,
            kind="product",
            model=args.llm_model,
            batch_size=min(args.llm_batch_size, 50),
            workers=args.llm_workers,
            cache_path=args.llm_attrs_products_cache_jsonl,
            max_output_tokens=args.llm_max_output_tokens,
            retries=args.llm_retries,
            throttle_seconds=args.llm_throttle_seconds,
            force=args.llm_force,
        )
        attribute_type_terms = _augment_terms_with_llm(
            terms=types,
            kind="attribute_type",
            model=args.llm_model,
            batch_size=min(args.llm_batch_size, 50),
            workers=args.llm_workers,
            cache_path=args.llm_attrs_types_cache_jsonl,
            max_output_tokens=args.llm_max_output_tokens,
            retries=args.llm_retries,
            throttle_seconds=args.llm_throttle_seconds,
            force=args.llm_force,
        )
        _augment_attributes_with_llm(
            objects=attr_objs,
            model=args.llm_model,
            batch_size=args.llm_batch_size,
            workers=args.llm_workers,
            product_terms=product_terms,
            attribute_type_terms=attribute_type_terms,
            cache_path=args.llm_attrs_cache_jsonl,
            max_output_tokens=args.llm_max_output_tokens,
            retries=args.llm_retries,
            throttle_seconds=args.llm_throttle_seconds,
            force=args.llm_force,
        )

    client = make_weaviate_client(
        weaviate_url=args.weaviate_url,
        weaviate_api_key=args.weaviate_api_key,
        grpc_port=args.grpc_port,
    )
    try:
        ensure_collections(client, recreate=args.recreate)
        batch_import(client, name="HS6", objects=hs6_objs, batch_size=args.batch_size)
        batch_import(client, name="HSCountry", objects=hs_country_objs, batch_size=args.batch_size)
        batch_import(client, name="Attributes", objects=attr_objs, batch_size=args.batch_size)
    finally:
        client.close()

    _eprint("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
