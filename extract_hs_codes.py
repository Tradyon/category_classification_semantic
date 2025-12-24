import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import dotenv
from openai import OpenAI

CLARIFAI_BASE_URL = "https://api.clarifai.com/v2/ext/openai/v1"
DEFAULT_DEEPSEEK_OCR_MODEL = "https://clarifai.com/deepseek-ai/deepseek-ocr/models/DeepSeek-OCR/versions/86b122666c2548f88d04dd998ccfbd70"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

HS_CODE_RE = re.compile(r"(?<!\d)(\d{2,4}(?:\.\d{2,4}){1,4})(?!\d)")

LLM_NORMALIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "hs_code": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["hs_code", "description"],
            },
        }
    },
    "required": ["rows"],
}

LLM_NORMALIZE_UPDATES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "updates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "hs_code": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["index", "hs_code", "description"],
            },
        }
    },
    "required": ["updates"],
}


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _add_token_usage(tally: dict[str, int], usage: dict[str, int] | None) -> None:
    if not usage:
        return
    for k, v in usage.items():
        if v is None:
            continue
        tally[k] = int(tally.get(k, 0)) + int(v)


def _extract_openai_usage(resp: Any) -> dict[str, int] | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None

    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)

    out: dict[str, int] = {}
    if prompt is not None:
        out["prompt_tokens"] = int(prompt)
    if completion is not None:
        out["completion_tokens"] = int(completion)
    if total is not None:
        out["total_tokens"] = int(total)
    return out or None


def load_env() -> None:
    dotenv.load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def load_clarifai_pat() -> str:
    load_env()
    pat = os.getenv("CLARIFAI_PAT") or os.getenv("CLARIFAI_API_KEY") or os.getenv("CLARIFAI_TOKEN")
    if not pat:
        raise RuntimeError(
            "Missing Clarifai credentials. Set `CLARIFAI_PAT` (preferred) or "
            "`CLARIFAI_API_KEY`/`CLARIFAI_TOKEN` in your environment or in `.env`."
        )
    return pat


def make_client() -> OpenAI:
    return OpenAI(base_url=CLARIFAI_BASE_URL, api_key=load_clarifai_pat())


def make_text_llm_client(model_name: str) -> OpenAI:
    load_env()
    model_lower = model_name.lower()
    if model_lower.startswith(("gemini", "models/gemini", "gemma", "models/gemma")):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing `GOOGLE_API_KEY` for Gemini model.")
        return OpenAI(base_url=GOOGLE_OPENAI_BASE_URL, api_key=api_key)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing `OPENROUTER_API_KEY` for non-Gemini model.")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"rows": obj}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("LLM response was not valid JSON object")


def call_text_llm_json(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    max_tokens: int,
    timeout: float,
    retries: int,
    token_tally: dict[str, int] | None = None,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        },
    }

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            if token_tally is not None:
                _add_token_usage(token_tally, _extract_openai_usage(resp))
            choice0 = resp.choices[0]
            finish_reason = getattr(choice0, "finish_reason", None)
            if finish_reason == "length":
                raise ValueError(
                    "LLM response truncated (finish_reason=length). Increase --llm-max-tokens "
                    "or use --llm-mode targeted."
                )
            content = str(getattr(getattr(choice0, "message", None), "content", "") or "")
            return _extract_json_object(content)
        except TypeError as e:
            last_err = e
            kwargs["response_format"] = {"type": "json_object"}
        except Exception as e:  # noqa: BLE001
            last_err = e
            maybe_rf = "response_format" in str(e).lower() or "json_schema" in str(e).lower()
            if maybe_rf and kwargs.get("response_format", {}).get("type") == "json_schema":
                kwargs["response_format"] = {"type": "json_object"}
            elif attempt >= retries:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"Text LLM request failed after {retries + 1} attempts") from last_err


def normalize_rows_with_llm(
    *,
    llm_client: OpenAI,
    llm_model: str,
    country: str | None,
    raw_text: str,
    rows: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    retries: int,
    token_tally: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    if not rows:
        return rows

    input_rows = [{"hs_code": r.get("hs_code", ""), "description": r.get("description", "")} for r in rows]
    raw_trimmed = (raw_text or "")[:8000]

    prompt = (
        "You are post-processing OCR output from HS code PDFs.\n"
        "Goal: For EACH input row, produce a FULL goods description by inheriting context from the\n"
        "closest applicable parent heading/subheading rows (based on HS code prefix, ignoring dots).\n"
        "Rules:\n"
        "- Do NOT add/remove/reorder rows.\n"
        "- Keep `hs_code` EXACTLY the same as input (same order).\n"
        "- Remove non-description noise (rates/duties/units like 'kg', 'Free', '%', '¢/kg', etc).\n"
        "- If a row is a refinement like 'Peas (Pisum sativum)' under heading 'Dried leguminous vegetables...',\n"
        "  output 'Dried leguminous vegetables... Peas (Pisum sativum)'.\n"
        "- For country-specific statistical suffix rows, the >8 digit codes are already formed; just generate\n"
        "  a full goods description for that code by inheriting from parents.\n"
        f"Country: {(country or '')}\n\n"
        "OCR text (for context):\n"
        f"{raw_trimmed}\n\n"
        "Input rows (JSON):\n"
        f"{json.dumps({'rows': input_rows}, ensure_ascii=False)}\n\n"
        "Return ONLY valid JSON matching schema: {\"rows\":[{\"hs_code\":string,\"description\":string}]}.\n"
    )

    obj = call_text_llm_json(
        client=llm_client,
        model=llm_model,
        prompt=prompt,
        schema=LLM_NORMALIZE_SCHEMA,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        token_tally=token_tally,
    )
    out_rows = obj.get("rows")
    if not isinstance(out_rows, list) or len(out_rows) != len(input_rows):
        raise ValueError("LLM output row count mismatch")

    normalized: list[dict[str, str]] = []
    for i, out in enumerate(out_rows):
        if not isinstance(out, dict):
            raise ValueError("LLM output rows must be objects")
        hs_code = str(out.get("hs_code", "")).strip()
        desc = str(out.get("description", "")).strip()
        if hs_code != input_rows[i]["hs_code"]:
            raise ValueError("LLM changed hs_code values/order")
        normalized.append({"hs_code": hs_code, "description": _normalize_desc(desc)})

    return normalized


def _should_llm_rewrite_row(hs_code: str, description: str, *, country: str | None) -> bool:
    if country and _code_digit_length(hs_code) > 8:
        return True
    if country and _code_digit_length(hs_code) == 4:
        return True

    desc = _normalize_desc(description)
    if not desc:
        return True
    if re.fullmatch(r"(?i)other:?\.?", desc):
        return True
    if _looks_like_rate(desc):
        return True

    words = desc.split()
    if len(words) <= 3:
        return True
    if len(desc) < 25:
        return True
    return False


def normalize_rows_with_llm_updates(
    *,
    llm_client: OpenAI,
    llm_model: str,
    country: str | None,
    raw_text: str,
    rows: list[dict[str, str]],
    target_indices: list[int],
    max_tokens: int,
    timeout: float,
    retries: int,
    token_tally: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    if not rows or not target_indices:
        return rows

    targets = [
        {
            "index": i,
            "hs_code": rows[i].get("hs_code", ""),
            "description": rows[i].get("description", ""),
        }
        for i in target_indices
    ]
    raw_trimmed = (raw_text or "")[:8000]
    context_rows = [{"hs_code": r.get("hs_code", ""), "description": r.get("description", "")} for r in rows]

    prompt = (
        "You are post-processing OCR output from HS code PDFs.\n"
        "Goal: Generate improved FULL goods descriptions for ONLY the target rows below by inheriting\n"
        "context from parent heading/subheading rows (based on HS code prefix, ignoring dots).\n"
        "Rules:\n"
        "- Do NOT create new codes.\n"
        "- Return updates ONLY for the provided target indices.\n"
        "- Keep `index` and `hs_code` EXACTLY the same.\n"
        "- Remove non-description noise (rates/duties/units like 'kg', 'Free', '%', '¢/kg', etc).\n"
        f"Country: {(country or '')}\n\n"
        "OCR text (for context):\n"
        f"{raw_trimmed}\n\n"
        "All rows (context, JSON):\n"
        f"{json.dumps({'rows': context_rows}, ensure_ascii=False)}\n\n"
        "Target rows to update (JSON):\n"
        f"{json.dumps({'targets': targets}, ensure_ascii=False)}\n\n"
        "Return ONLY valid JSON matching schema: {\"updates\":[{\"index\":int,\"hs_code\":string,\"description\":string}]}.\n"
    )

    obj = call_text_llm_json(
        client=llm_client,
        model=llm_model,
        prompt=prompt,
        schema=LLM_NORMALIZE_UPDATES_SCHEMA,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        token_tally=token_tally,
    )
    updates = obj.get("updates")
    if not isinstance(updates, list):
        raise ValueError("LLM output missing updates list")

    allowed = set(target_indices)
    seen: set[int] = set()
    updated_rows = [dict(r) for r in rows]
    for upd in updates:
        if not isinstance(upd, dict):
            raise ValueError("LLM updates must be objects")
        idx = upd.get("index")
        if not isinstance(idx, int) or idx not in allowed:
            raise ValueError("LLM returned unexpected update index")
        if idx in seen:
            raise ValueError("LLM returned duplicate update index")
        seen.add(idx)
        hs_code = str(upd.get("hs_code", "")).strip()
        if hs_code != rows[idx].get("hs_code", ""):
            raise ValueError("LLM changed hs_code")
        desc = _normalize_desc(str(upd.get("description", "")).strip())
        updated_rows[idx]["description"] = desc

    if seen != allowed:
        missing = sorted(allowed - seen)
        raise ValueError(f"LLM did not return updates for indices: {missing}")

    return updated_rows


def list_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.pdf") if p.is_file())


def get_pdf_page_count(pdf_path: Path) -> int:
    try:
        out = subprocess.check_output(
            ["pdfinfo", str(pdf_path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as e:
        raise RuntimeError("`pdfinfo` not found. Install poppler-utils.") from e
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Unable to determine page count for {pdf_path}")


def render_pdf_page_jpeg(pdf_path: Path, page: int, out_prefix: Path, dpi: int) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(dpi),
                "-jpeg",
                "-singlefile",
                str(pdf_path),
                str(out_prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as e:
        raise RuntimeError("`pdftoppm` not found. Install poppler-utils.") from e
    out_path = out_prefix.with_suffix(".jpg")
    if not out_path.exists():
        raise RuntimeError(f"Expected render output not found: {out_path}")
    return out_path


def _image_to_data_url(image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def call_deepseek_ocr(
    *,
    client: OpenAI,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    retries: int,
    token_tally: dict[str, int] | None = None,
) -> str:
    data_url = _image_to_data_url(image_path)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are an OCR and table extraction engine."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
            )
            if token_tally is not None:
                _add_token_usage(token_tally, _extract_openai_usage(resp))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - surfacing upstream API errors is useful
            last_err = e
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek-OCR request failed after {retries + 1} attempts") from last_err


def _is_md_delimiter_row(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        stripped = cell.strip().replace(" ", "")
        if not stripped:
            return False
        if not re.fullmatch(r":?-{3,}:?", stripped):
            return False
    return True


def _normalize_hs_code(code: str) -> str:
    code = code.strip().replace(" ", "")
    code = re.sub(r"[^0-9.]", "", code)
    return code.strip(".")


def _digits_only(code: str) -> str:
    return re.sub(r"\D", "", code)


def _code_digit_length(code: str) -> int:
    return len(_digits_only(code))


def _clean_prefix_for_join(prefix: str) -> str:
    prefix = _normalize_desc(prefix)
    return prefix.rstrip(" .:;,-")


def _join_descriptions(prefix: str, desc: str) -> str:
    prefix_clean = _clean_prefix_for_join(prefix)
    desc = _normalize_desc(desc)
    if not prefix_clean:
        return desc
    if not desc:
        return prefix_clean
    if prefix_clean.lower() in desc.lower():
        return desc
    if desc.lower() in prefix_clean.lower():
        return prefix_clean
    return _normalize_desc(f"{prefix_clean} {desc}")


def _looks_like_rate(s: str) -> bool:
    lowered = s.lower()
    return ("free" in lowered) or ("%" in s) or ("¢" in s)


def _is_meaningful_prefix(desc: str) -> bool:
    desc = _normalize_desc(re.sub(r"\.{2,}", " ", desc))
    if len(desc) < 8:
        return False
    if _looks_like_rate(desc):
        return False
    if re.fullmatch(r"(?i)other:?", desc):
        return False
    if not re.search(r"[A-Za-z]", desc):
        return False
    return True


def _is_heading_code(hs_code: str) -> bool:
    return bool(re.fullmatch(r"\d{2}\.\d{2}", hs_code))


def _extract_hs_code_from_cell(cell: str, *, allow_heading: bool) -> str | None:
    cleaned = re.sub(r"[^0-9.]", "", cell)
    m = HS_CODE_RE.search(cleaned)
    if m:
        return _normalize_hs_code(m.group(1))
    if allow_heading and re.fullmatch(r"\d{4}", cleaned):
        return cleaned
    return None


def _normalize_desc(desc: str) -> str:
    desc = re.sub(r"\s+", " ", desc).strip()
    return desc


def _pick_description_from_cells(cells: list[str], code_idx: int) -> str:
    def has_letters(s: str) -> bool:
        return bool(re.search(r"[A-Za-z]", s))

    other_cells = [c.strip() for i, c in enumerate(cells) if i != code_idx and c.strip()]
    if not other_cells:
        return ""
    letter_cells = [c for c in other_cells if has_letters(c) and not _looks_like_rate(c)]
    if letter_cells:
        return max(letter_cells, key=len)
    letter_cells = [c for c in other_cells if has_letters(c)]
    pool = letter_cells or other_cells
    return max(pool, key=len)


def extract_hs_rows_from_markdown(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_table: list[str] = []
    tables: list[list[str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.endswith("|") and line.count("|") >= 2:
            current_table.append(line)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []
    if current_table:
        tables.append(current_table)

    for table in tables:
        parsed_rows: list[list[str]] = []
        for line in table:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if _is_md_delimiter_row(cells):
                continue
            parsed_rows.append(cells)

        if not parsed_rows:
            continue

        header_idx: int | None = None
        desc_idx: int | None = None
        code_hint_idx: int | None = None
        suffix_idx: int | None = None
        for i, header_cells in enumerate(parsed_rows[:3]):
            for j, cell in enumerate(header_cells):
                if "description" in cell.lower():
                    header_idx = i
                    desc_idx = j
                    break
            if header_idx is not None:
                break

        if header_idx is not None:
            header_cells_lower = [c.lower() for c in parsed_rows[header_idx]]
            for j, cell in enumerate(header_cells_lower):
                if any(k in cell for k in ("h.s", "hs", "subheading", "heading", "code")):
                    code_hint_idx = j
                    break
            for j, cell in enumerate(header_cells_lower):
                if ("suffix" in cell) or ("stat" in cell and "suf" in cell):
                    suffix_idx = j
                    break

        start_idx = (header_idx + 1) if header_idx is not None else 0
        last_desc: str | None = None
        last_base_code_8: str | None = None
        for cells in parsed_rows[start_idx:]:
            if not any(c.strip() for c in cells):
                continue

            hs_code: str | None = None
            code_idx = -1
            if code_hint_idx is not None and code_hint_idx < len(cells):
                hs_code = _extract_hs_code_from_cell(cells[code_hint_idx], allow_heading=True)
                if hs_code:
                    code_idx = code_hint_idx

            if not hs_code:
                for idx, cell in enumerate(cells):
                    hs_code = _extract_hs_code_from_cell(cell, allow_heading=(idx == 0))
                    if hs_code:
                        code_idx = idx
                        break

            suffix_val: str | None = None
            if suffix_idx is not None and suffix_idx < len(cells):
                suffix_digits = re.sub(r"\D", "", cells[suffix_idx])
                if suffix_digits:
                    if len(suffix_digits) == 1:
                        suffix_digits = f"0{suffix_digits}"
                    if len(suffix_digits) == 2:
                        suffix_val = suffix_digits

            if not hs_code:
                if suffix_val and last_base_code_8:
                    hs_code = f"{last_base_code_8}{suffix_val}"
                    code_idx = -1
                else:
                    hs_code = None

            if hs_code and _code_digit_length(hs_code) == 8:
                last_base_code_8 = hs_code
            elif hs_code and _code_digit_length(hs_code) <= 6:
                last_base_code_8 = None

            if hs_code and suffix_val and _code_digit_length(hs_code) == 8:
                hs_code = f"{hs_code}{suffix_val}"

            if not hs_code:
                if desc_idx is not None and desc_idx < len(cells):
                    continuation = _normalize_desc(cells[desc_idx])
                    other_nonempty = [
                        c.strip()
                        for i, c in enumerate(cells)
                        if i != desc_idx and c.strip()
                    ]
                    if (
                        continuation
                        and rows
                        and not other_nonempty
                        and not re.fullmatch(r"(?i)other:?", continuation)
                    ):
                        rows[-1]["description"] = _normalize_desc(
                            f"{rows[-1]['description']} {continuation}".strip()
                        )
                else:
                    nonempty = [c.strip() for c in cells if c.strip()]
                    if len(nonempty) <= 1 and nonempty and rows:
                        rows[-1]["description"] = _normalize_desc(
                            f"{rows[-1]['description']} {nonempty[0]}".strip()
                        )
                continue

            desc = ""
            if desc_idx is not None and desc_idx < len(cells):
                desc = cells[desc_idx].strip()
            if not desc and last_desc:
                desc = last_desc
            if not desc:
                desc = _pick_description_from_cells(cells, code_idx)

            desc = _normalize_desc(re.sub(r"\.{2,}", " ", desc))
            if desc:
                last_desc = desc
            rows.append({"hs_code": hs_code, "description": desc})

    return rows


def augment_rows_with_hierarchy(
    rows: list[dict[str, str]],
    desc_by_digits: dict[str, str],
) -> list[dict[str, str]]:
    augmented: list[dict[str, str]] = []
    for row in rows:
        hs_code = row.get("hs_code", "").strip()
        desc = row.get("description", "").strip()
        digits = _digits_only(hs_code)

        prefix: str | None = None
        for k in range(len(digits) - 1, 1, -1):
            cand = digits[:k]
            cand_desc = desc_by_digits.get(cand)
            if cand_desc and _is_meaningful_prefix(cand_desc):
                prefix = cand_desc
                break

        if prefix and desc:
            desc = _join_descriptions(prefix, desc)

        out_row = {"hs_code": hs_code, "description": desc}
        augmented.append(out_row)

        if digits and desc and _is_heading_code(hs_code) and _is_meaningful_prefix(desc):
            desc_by_digits[digits] = desc

    return augmented


def update_hierarchy_context_from_rows(
    rows: list[dict[str, str]],
    desc_by_digits: dict[str, str],
) -> None:
    for row in rows:
        hs_code = row.get("hs_code", "").strip()
        desc = row.get("description", "").strip()
        digits = _digits_only(hs_code)
        if digits and desc and _is_heading_code(hs_code) and _is_meaningful_prefix(desc):
            desc_by_digits[digits] = desc


def extract_hs_rows(text: str) -> list[dict[str, str]]:
    rows = extract_hs_rows_from_markdown(text)
    if rows:
        return rows

    def strip_emphasis(s: str) -> str:
        return re.sub(r"[*_]+", "", s).strip()

    def strip_leading_dashes(s: str) -> str:
        return re.sub(r"^[-–—]{1,3}\s*", "", s.strip()).strip()

    extracted: list[dict[str, str]] = []
    prefix: str | None = None
    prefix_in_progress = False

    for raw_line in text.splitlines():
        line = strip_emphasis(raw_line.strip())
        if not line:
            continue

        m = re.match(
            r"^(?P<code>\d{2,4}(?:\.\d{2,4}){1,4})\s*(?P<desc>.*)$",
            line,
        )
        if m:
            code = _normalize_hs_code(m.group("code"))
            raw_desc = m.group("desc")
            desc = _normalize_desc(strip_leading_dashes(raw_desc))
            if prefix and raw_desc.lstrip().startswith("--") and desc:
                desc = _normalize_desc(f"{prefix} - {desc}")

            extracted.append({"hs_code": code, "description": desc})
            prefix_in_progress = False
            if not raw_desc.lstrip().startswith("--"):
                prefix = None
            continue

        if line.lstrip().startswith(("-", "–", "—")):
            bullet = strip_leading_dashes(line).rstrip(":").strip()
            if bullet:
                prefix = bullet
                prefix_in_progress = True
            continue

        if prefix_in_progress and prefix:
            prefix = _normalize_desc(f"{prefix} {line}".rstrip(":").strip())
            continue

        if extracted:
            extracted[-1]["description"] = _normalize_desc(
                f"{extracted[-1]['description']} {line}".strip()
            )
    return extracted


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_processed_pages(path: Path) -> set[tuple[str, int]]:
    processed: set[tuple[str, int]] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pdf = obj.get("pdf")
            page = obj.get("page")
            if isinstance(pdf, str) and isinstance(page, int):
                processed.add((pdf, page))
    return processed


def write_rows_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pdf", "page", "country", "hs_code", "description"]
    has_header = False
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            first_line = f.readline().rstrip("\r\n")
        expected = ",".join(fieldnames)
        if first_line:
            has_header = True
        if has_header and first_line != expected:
            raise RuntimeError(
                f"Existing CSV header mismatch in {csv_path}. Expected: {expected!r} "
                f"but found: {first_line!r}. Use a new --rows-csv path or delete the file."
            )
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_country(pdf_path: Path) -> str | None:
    parts_lower = [p.lower() for p in pdf_path.parts]
    if "country specific" not in parts_lower:
        return None

    name_lower = pdf_path.name.lower()
    if "hts" in name_lower:
        return "US"

    try:
        text = subprocess.check_output(
            ["pdftotext", "-f", "1", "-l", "1", str(pdf_path), "-"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return "UNKNOWN"

    text_lower = text.lower()
    if "united states" in text_lower:
        return "US"
    return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract HS codes + goods descriptions from PDFs using DeepSeek-OCR (Clarifai)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("HS Code"),
        help="PDF file or directory to scan recursively (default: HS Code/)",
    )
    parser.add_argument(
        "--pages-jsonl",
        type=Path,
        default=Path("outputs/hs_pages.jsonl"),
        help="Page-level JSONL output (default: outputs/hs_pages.jsonl)",
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=Path("outputs/hs_rows.csv"),
        help="Row-level CSV output (default: outputs/hs_rows.csv)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hs_extract/pages"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--model", type=str, default=DEFAULT_DEEPSEEK_OCR_MODEL)
    parser.add_argument("--max-pdfs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no limit (per PDF)")
    parser.add_argument("--from-page", type=int, default=1)
    parser.add_argument("--to-page", type=int, default=0, help="0 means last page")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing --pages-jsonl (default: True)",
    )
    parser.add_argument(
        "--llm-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an LLM to generate full goods descriptions (default: True)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model for description generation (OpenRouter or Gemini).",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="targeted",
        choices=["targeted", "all"],
        help="LLM post-processing mode: targeted (rewrite short/noisy rows) or all (rewrite every row).",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    args = parser.parse_args()

    client = make_client()
    llm_client = make_text_llm_client(args.llm_model) if args.llm_normalize else None
    pdfs = list_pdfs(args.input)
    if not pdfs:
        raise RuntimeError(f"No PDFs found under: {args.input}")
    if args.max_pdfs > 0:
        pdfs = pdfs[: args.max_pdfs]

    processed = load_processed_pages(args.pages_jsonl) if args.resume else set()

    prompt_primary = (
        "Extract the table(s) from this page. If the page contains HS/H.S. codes and goods "
        "descriptions, make sure those appear in the extracted tables. Return the extracted "
        "tables in Markdown."
    )
    prompt_fallback = "Extract all text from this page."

    total_pages_done = 0
    ocr_token_totals: dict[str, int] = {}
    llm_token_totals: dict[str, int] = {}
    for pdf_path in pdfs:
        country = infer_country(pdf_path)
        desc_by_digits: dict[str, str] = {}
        page_count = get_pdf_page_count(pdf_path)
        start = max(1, args.from_page)
        end = page_count if args.to_page <= 0 else min(page_count, args.to_page)
        if args.max_pages > 0:
            end = min(end, start + args.max_pages - 1)

        _eprint(f"{pdf_path} ({page_count} pages) -> processing {start}-{end}")
        for page in range(start, end + 1):
            pdf_key = str(pdf_path)
            if (pdf_key, page) in processed:
                continue

            out_prefix = args.cache_dir / pdf_path.stem / f"page_{page:04d}"
            image_path = out_prefix.with_suffix(".jpg")
            if not image_path.exists():
                image_path = render_pdf_page_jpeg(
                    pdf_path=pdf_path, page=page, out_prefix=out_prefix, dpi=args.dpi
                )

            page_ocr_usage: dict[str, int] = {}
            page_llm_usage: dict[str, int] = {}

            text = call_deepseek_ocr(
                client=client,
                model=args.model,
                image_path=image_path,
                prompt=prompt_primary,
                max_tokens=1800,
                retries=3,
                token_tally=page_ocr_usage,
            )
            if not text.strip():
                text = call_deepseek_ocr(
                    client=client,
                    model=args.model,
                    image_path=image_path,
                    prompt=prompt_fallback,
                    max_tokens=1800,
                    retries=3,
                    token_tally=page_ocr_usage,
                )

            rows = extract_hs_rows(text)
            rows = augment_rows_with_hierarchy(rows, desc_by_digits)
            if args.llm_normalize and llm_client and rows:
                try:
                    if args.llm_mode == "all":
                        rows = normalize_rows_with_llm(
                            llm_client=llm_client,
                            llm_model=args.llm_model,
                            country=country,
                            raw_text=text,
                            rows=rows,
                            max_tokens=args.llm_max_tokens,
                            timeout=args.llm_timeout,
                            retries=args.llm_retries,
                            token_tally=page_llm_usage,
                        )
                    else:
                        target_indices = [
                            i
                            for i, r in enumerate(rows)
                            if _should_llm_rewrite_row(
                                r.get("hs_code", ""),
                                r.get("description", ""),
                                country=country,
                            )
                        ]
                        if target_indices:
                            rows = normalize_rows_with_llm_updates(
                                llm_client=llm_client,
                                llm_model=args.llm_model,
                                country=country,
                                raw_text=text,
                                rows=rows,
                                target_indices=target_indices,
                                max_tokens=args.llm_max_tokens,
                                timeout=args.llm_timeout,
                                retries=args.llm_retries,
                                token_tally=page_llm_usage,
                            )
                    update_hierarchy_context_from_rows(rows, desc_by_digits)
                except Exception as e:  # noqa: BLE001
                    cause = getattr(e, "__cause__", None)
                    if cause:
                        _eprint(f"  page {page}: LLM normalize failed: {cause}")
                    else:
                        _eprint(f"  page {page}: LLM normalize failed: {e}")

            _add_token_usage(ocr_token_totals, page_ocr_usage)
            _add_token_usage(llm_token_totals, page_llm_usage)
            page_total_usage: dict[str, int] = {}
            _add_token_usage(page_total_usage, page_ocr_usage)
            _add_token_usage(page_total_usage, page_llm_usage)
            page_record = {
                "pdf": pdf_key,
                "page": page,
                "dpi": args.dpi,
                "model": args.model,
                "country": country,
                "usage": {"ocr": page_ocr_usage, "llm": page_llm_usage, "total": page_total_usage},
                "raw": text,
                "rows": rows,
            }
            append_jsonl(args.pages_jsonl, page_record)
            processed.add((pdf_key, page))

            if rows:
                write_rows_csv(
                    args.rows_csv,
                    [
                        {
                            "pdf": pdf_key,
                            "page": page,
                            "country": (country if _code_digit_length(r["hs_code"]) > 8 else ""),
                            **r,
                        }
                        for r in rows
                    ],
                )
            total_pages_done += 1
            if page_total_usage and page_total_usage.get("total_tokens", 0):
                _eprint(
                    f"  page {page}: {len(rows)} rows | "
                    f"tokens ocr p={page_ocr_usage.get('prompt_tokens', 0)} "
                    f"c={page_ocr_usage.get('completion_tokens', 0)} "
                    f"t={page_ocr_usage.get('total_tokens', 0)} | "
                    f"llm p={page_llm_usage.get('prompt_tokens', 0)} "
                    f"c={page_llm_usage.get('completion_tokens', 0)} "
                    f"t={page_llm_usage.get('total_tokens', 0)} | "
                    f"run t={ocr_token_totals.get('total_tokens', 0) + llm_token_totals.get('total_tokens', 0)}"
                )
            else:
                _eprint(f"  page {page}: {len(rows)} rows")

    _eprint(f"Done. Processed {total_pages_done} pages.")
    if ocr_token_totals or llm_token_totals:
        total_tokens = ocr_token_totals.get("total_tokens", 0) + llm_token_totals.get("total_tokens", 0)
        total_prompt = ocr_token_totals.get("prompt_tokens", 0) + llm_token_totals.get("prompt_tokens", 0)
        total_completion = ocr_token_totals.get("completion_tokens", 0) + llm_token_totals.get("completion_tokens", 0)
        _eprint(
            "Token totals: "
            f"ocr p={ocr_token_totals.get('prompt_tokens', 0)} "
            f"c={ocr_token_totals.get('completion_tokens', 0)} "
            f"t={ocr_token_totals.get('total_tokens', 0)} | "
            f"llm p={llm_token_totals.get('prompt_tokens', 0)} "
            f"c={llm_token_totals.get('completion_tokens', 0)} "
            f"t={llm_token_totals.get('total_tokens', 0)} | "
            f"total p={total_prompt} c={total_completion} t={total_tokens}"
        )
    _eprint(f"Wrote: {args.pages_jsonl}")
    _eprint(f"Wrote: {args.rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
