import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm


class HSRow(BaseModel):
    hs_code: str = Field(
        ...,
        description=(
            "HS code exactly as shown in the PDF (keep dots if present). "
            "May be 4-digit headings like '07.13', regular codes like '0713.10', "
            "or country-specific codes like '0701.90.5010'."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Goods description for the HS code, including inherited context from the closest "
            "parent heading/subheading when applicable. Exclude duty rates/units."
        ),
    )


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _add_token_usage(tally: dict[str, int], usage: dict[str, int] | None) -> None:
    if not usage:
        return
    for k, v in usage.items():
        if v is None:
            continue
        tally[k] = int(tally.get(k, 0)) + int(v)


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


def _normalize_hs_code(code: str) -> str:
    code = (code or "").strip().replace(" ", "")
    code = re.sub(r"[^0-9.]", "", code)
    return code.strip(".")


def _strip_leading_bullets(desc: str) -> str:
    return re.sub(r"^[-–—]{1,3}\s*", "", (desc or "").strip()).strip()


def _normalize_desc(desc: str) -> str:
    desc = re.sub(r"\s+", " ", (desc or "")).strip()
    return _strip_leading_bullets(desc)


def _looks_like_rate(s: str) -> bool:
    lowered = (s or "").lower()
    return ("free" in lowered) or ("%" in (s or "")) or ("¢" in (s or ""))


def _is_meaningful_prefix(desc: str) -> bool:
    desc = _normalize_desc(re.sub(r"\.{2,}", " ", desc or ""))
    if len(desc) < 8:
        return False
    if _looks_like_rate(desc):
        return False
    if re.fullmatch(r"(?i)other:?", desc):
        return False
    if not re.search(r"[A-Za-z]", desc):
        return False
    return True


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


def _should_store_as_prefix(hs_code: str, raw_desc: str) -> bool:
    digits_len = _code_digit_length(hs_code)
    if digits_len not in {4, 6, 8}:
        return False
    return _is_meaningful_prefix(raw_desc)


def augment_rows_with_hierarchy(
    rows: list[dict[str, str]],
    desc_by_digits: dict[str, str],
) -> list[dict[str, str]]:
    augmented: list[dict[str, str]] = []
    for row in rows:
        hs_code = _normalize_hs_code(row.get("hs_code", ""))
        raw_desc = _normalize_desc(row.get("description", ""))
        desc = raw_desc
        digits = _digits_only(hs_code)

        prefix: str | None = None
        for k in range(len(digits) - 1, 1, -1):
            cand = digits[:k]
            cand_desc = desc_by_digits.get(cand)
            if cand_desc and _is_meaningful_prefix(cand_desc):
                prefix = cand_desc
                break

        if prefix:
            desc = _join_descriptions(prefix, desc)

        augmented.append({"hs_code": hs_code, "description": desc})

        if digits and desc and _should_store_as_prefix(hs_code, raw_desc):
            desc_by_digits[digits] = desc

    return augmented


def load_env() -> None:
    dotenv.load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def load_google_api_key() -> str:
    load_env()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing `GOOGLE_API_KEY` in environment or `.env`.")
    return key


def make_client() -> genai.Client:
    return genai.Client(api_key=load_google_api_key())


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
            if isinstance(pdf, str) and isinstance(page, int) and not obj.get("error"):
                processed.add((pdf, page))
    return processed


def load_rows_by_pdf(path: Path) -> dict[str, dict[int, list[dict[str, str]]]]:
    by_pdf: dict[str, dict[int, list[dict[str, str]]]] = {}
    if not path.exists():
        return by_pdf

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("error"):
                continue

            pdf = obj.get("pdf")
            page = obj.get("page")
            rows = obj.get("rows")
            if not isinstance(pdf, str) or not isinstance(page, int) or not isinstance(rows, list):
                continue

            cleaned_rows: list[dict[str, str]] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                hs_code = str(r.get("hs_code", "") or "")
                description = str(r.get("description", "") or "")
                if hs_code.strip():
                    cleaned_rows.append({"hs_code": hs_code, "description": description})

            if cleaned_rows:
                by_pdf.setdefault(pdf, {})[page] = cleaned_rows

    return by_pdf


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


def _digits_only(code: str) -> str:
    return re.sub(r"\D", "", code or "")


def _code_digit_length(code: str) -> int:
    return len(_digits_only(code))


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


def get_file_hash(file_path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - used only for checkpointing, not security
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gemini_extract_hs_rows(
    *,
    client: genai.Client,
    models: list[str],
    image_bytes: bytes,
    country: str | None,
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict[str, str]], str, str, dict[str, int]]:
    prompt = (
        "Extract HS/H.S. codes and their goods descriptions from this page.\n"
        "Return ONLY structured JSON: an array of {hs_code, description}.\n"
        "Rules:\n"
        "- Keep dots in hs_code if present.\n"
        "- Include heading rows (e.g. '07.13') if they appear; these are required for hierarchy.\n"
        "- IMPORTANT: If the page has a Stat. Suffix / statistical suffix column (2 digits like 20, 40, 10, 60),\n"
        "  output the full 10-digit code by appending the suffix to the base 8-digit code (e.g. 0701.90.5010).\n"
        "- Do NOT output duplicate base codes for different suffix rows; each output row must have a unique full code.\n"
        "- description MUST be self-contained and meaningful.\n"
        "  Never output a bare/generic description like 'Other', 'Seed', or a single word.\n"
        "  If the printed row is short/generic, prepend the closest parent heading/subheading text from the page.\n"
        "  Example:\n"
        "    07.13  Dried leguminous vegetables, shelled, whether or not skinned or split\n"
        "    0713.10  Peas (Pisum sativum)\n"
        "  Output for 0713.10:\n"
        "    'Dried leguminous vegetables, shelled, whether or not skinned or split Peas (Pisum sativum)'\n"
        "- Merge continuation lines/rows without a code into the previous description.\n"
        "- Do NOT include duties, rates, or units (e.g. kg, Free, %, ¢/kg) in description.\n"
        "- Do NOT start descriptions with bullet characters like '-' or '–'.\n"
        f"Country (if applicable): {(country or '')}\n"
    )

    last_err: Exception | None = None
    model_list = [m.strip() for m in (models or []) if m and m.strip()]
    if not model_list:
        raise ValueError("No Gemini models provided")

    usage_total: dict[str, int] = {}
    for model in model_list:
        for attempt in range(retries + 1):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            ],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[HSRow],
                        max_output_tokens=max_output_tokens,
                    ),
                )

                raw_text = (resp.text or "").strip()
                _add_token_usage(usage_total, _extract_gemini_usage(resp))

                parsed = resp.parsed
                if isinstance(parsed, list):
                    return (
                        [r.model_dump() if isinstance(r, HSRow) else dict(r) for r in parsed],
                        raw_text,
                        model,
                        usage_total,
                    )

                if not raw_text:
                    raise ValueError("Empty Gemini response")

                try:
                    obj = json.loads(raw_text)
                except json.JSONDecodeError as e:
                    raise ValueError("Gemini returned invalid JSON") from e

                if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
                    obj = obj["rows"]

                if not isinstance(obj, list):
                    raise ValueError("Gemini JSON was not a list")

                rows = [HSRow.model_validate(item).model_dump() for item in obj]
                return rows, raw_text, model, usage_total
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= retries:
                    break
                time.sleep(2**attempt)

    raise RuntimeError(
        f"Gemini extraction failed after trying models: {', '.join(model_list)}"
    ) from last_err


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract HS codes + goods descriptions from PDFs using Gemini (direct API)."
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
        default=Path("outputs/hs_pages_gemini.jsonl"),
        help="Page-level JSONL output (default: outputs/hs_pages_gemini.jsonl)",
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=Path("outputs/hs_rows_gemini.csv"),
        help="Row-level CSV output (default: outputs/hs_rows_gemini.csv)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hs_extract/pages_gemini"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument(
        "--fallback-model",
        type=str,
        default="gemini-2.0-flash",
        help="Fallback model used if the primary model fails to return valid JSON.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--throttle-seconds", type=float, default=1.0)
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
    args = parser.parse_args()

    client = make_client()
    pdfs = list_pdfs(args.input)
    if not pdfs:
        raise RuntimeError(f"No PDFs found under: {args.input}")
    if args.max_pdfs > 0:
        pdfs = pdfs[: args.max_pdfs]

    processed = load_processed_pages(args.pages_jsonl) if args.resume else set()
    prior_rows_by_pdf = load_rows_by_pdf(args.pages_jsonl) if args.resume else {}
    token_totals: dict[str, int] = {}

    total_pages_to_process = 0
    for pdf_path in pdfs:
        pdf_key = str(pdf_path)
        page_count = get_pdf_page_count(pdf_path)
        start = max(1, args.from_page)
        end = page_count if args.to_page <= 0 else min(page_count, args.to_page)
        if args.max_pages > 0:
            end = min(end, start + args.max_pages - 1)
        for page in range(start, end + 1):
            if (pdf_key, page) not in processed:
                total_pages_to_process += 1

    total_pages_done = 0
    pbar = tqdm(total=total_pages_to_process, desc="Extract pages", unit="page", file=sys.stderr)
    for pdf_path in pdfs:
        country = infer_country(pdf_path)
        desc_by_digits: dict[str, str] = {}
        page_count = get_pdf_page_count(pdf_path)
        start = max(1, args.from_page)
        end = page_count if args.to_page <= 0 else min(page_count, args.to_page)
        if args.max_pages > 0:
            end = min(end, start + args.max_pages - 1)

        _eprint(f"{pdf_path} ({page_count} pages) -> processing {start}-{end}")
        pdf_key = str(pdf_path)
        prior_rows_by_page = prior_rows_by_pdf.get(pdf_key, {})
        if prior_rows_by_page and start > 1:
            for prev_page in sorted(p for p in prior_rows_by_page if p < start):
                augment_rows_with_hierarchy(prior_rows_by_page[prev_page], desc_by_digits)
        for page in range(start, end + 1):
            if (pdf_key, page) in processed:
                prev_rows = prior_rows_by_page.get(page)
                if prev_rows:
                    augment_rows_with_hierarchy(prev_rows, desc_by_digits)
                continue

            out_prefix = args.cache_dir / pdf_path.stem / f"page_{page:04d}"
            image_path = out_prefix.with_suffix(".jpg")
            if not image_path.exists():
                image_path = render_pdf_page_jpeg(
                    pdf_path=pdf_path, page=page, out_prefix=out_prefix, dpi=args.dpi
                )
            image_bytes = image_path.read_bytes()

            try:
                model_chain = [args.model]
                if args.fallback_model and args.fallback_model != args.model:
                    model_chain.append(args.fallback_model)

                rows, raw_text, used_model, usage = gemini_extract_hs_rows(
                    client=client,
                    models=model_chain,
                    image_bytes=image_bytes,
                    country=country,
                    max_output_tokens=args.max_output_tokens,
                    retries=args.retries,
                )
                rows = augment_rows_with_hierarchy(rows, desc_by_digits)
                error: str | None = None
            except Exception as e:  # noqa: BLE001
                rows, raw_text = [], ""
                error = str(e)
                _eprint(f"  page {page}: extraction failed: {error}")
                used_model = args.model
                usage = {}

            _add_token_usage(token_totals, usage)
            page_record = {
                "pdf": pdf_key,
                "page": page,
                "dpi": args.dpi,
                "model": used_model,
                "country": country,
                "usage": usage,
                "raw": raw_text,
                "error": error,
                "rows": rows,
            }
            append_jsonl(args.pages_jsonl, page_record)
            if not error:
                processed.add((pdf_key, page))

            if rows:
                write_rows_csv(
                    args.rows_csv,
                    [
                        {
                            "pdf": pdf_key,
                            "page": page,
                            "country": (country if _code_digit_length(r["hs_code"]) > 8 else ""),
                            "hs_code": _normalize_hs_code(r.get("hs_code") or ""),
                            "description": _normalize_desc(r.get("description") or ""),
                        }
                        for r in rows
                    ],
                )

            total_pages_done += 1
            pbar.set_description(f"{pdf_path.name}")
            pbar.update(1)
            pbar.set_postfix(rows=len(rows))
            if usage and any(usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                _eprint(
                    f"  page {page}: {len(rows)} rows | tokens p={usage.get('prompt_tokens', 0)} "
                    f"c={usage.get('completion_tokens', 0)} t={usage.get('total_tokens', 0)} | "
                    f"run t={token_totals.get('total_tokens', 0)}"
                )
            else:
                _eprint(f"  page {page}: {len(rows)} rows")
            if args.throttle_seconds > 0:
                time.sleep(args.throttle_seconds)

    pbar.close()
    _eprint(f"Done. Processed {total_pages_done} pages.")
    if token_totals:
        _eprint(
            "Token totals: "
            f"prompt={token_totals.get('prompt_tokens', 0)} "
            f"completion={token_totals.get('completion_tokens', 0)} "
            f"total={token_totals.get('total_tokens', 0)}"
        )
    _eprint(f"Wrote: {args.pages_jsonl}")
    _eprint(f"Wrote: {args.rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
