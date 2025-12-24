import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from weaviate.classes.config import Configure, DataType, Property
from weaviate.util import generate_uuid5

from weaviate_utils import (
    digits_only,
    infer_country_from_pdf_path,
    is_country_specific_pdf,
    make_weaviate_client,
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
                "text": f"{hs_code} {description}".strip(),
                "hs_code": hs_code,
                "description": description,
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
                "text": f"{country} {hs_code} {description}".strip(),
                "country": country,
                "hs_code": hs_code,
                "description": description,
                "digits": digits,
                "source_pdf": pdf,
                "page": page,
            }
    return list(best_by_key.values())


def load_attribute_triples(attrs_json_path: Path) -> list[dict[str, str]]:
    if not attrs_json_path.exists():
        raise FileNotFoundError(f"Missing attrs JSON: {attrs_json_path}")
    obj = json.loads(attrs_json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("attrs.json must be a JSON object mapping product -> schema")

    seen: set[str] = set()
    triples: list[dict[str, str]] = []
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
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
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
        for obj in objects:
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
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop and recreate the collections before indexing (default: False).",
    )
    args = parser.parse_args()

    hs_rows = load_hs_rows(args.hs_csv)
    hs6_objs = build_hs6_objects(hs_rows)
    hs_country_objs = build_hs_country_objects(hs_rows)
    attr_objs = load_attribute_triples(args.attrs_json)

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

