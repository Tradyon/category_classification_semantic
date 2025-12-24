import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dotenv
import weaviate


def load_env() -> None:
    dotenv.load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def digits_only(code: str) -> str:
    return re.sub(r"\D", "", code or "")


def is_country_specific_pdf(pdf_path: str) -> bool:
    parts_lower = [p.lower() for p in Path(pdf_path).parts]
    return "country specific" in parts_lower


def infer_country_from_pdf_path(pdf_path: str) -> str:
    name_lower = Path(pdf_path).name.lower()
    if "hts" in name_lower:
        return "US"
    return "UNKNOWN"


def make_weaviate_client(
    *,
    weaviate_url: str | None = None,
    weaviate_api_key: str | None = None,
    grpc_port: int | None = None,
) -> weaviate.WeaviateClient:
    load_env()
    url = weaviate_url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = weaviate_api_key or os.getenv("WEAVIATE_API_KEY", "")
    grpc_port_val = grpc_port
    if grpc_port_val is None:
        grpc_env = os.getenv("WEAVIATE_GRPC_PORT", "")
        grpc_port_val = int(grpc_env) if grpc_env.strip().isdigit() else 50051

    parsed = urlparse(url)
    http_host = parsed.hostname or "localhost"
    http_secure = parsed.scheme == "https"
    http_port = parsed.port or (443 if http_secure else 8080)

    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=int(http_port),
        http_secure=http_secure,
        grpc_host=http_host,
        grpc_port=int(grpc_port_val),
        grpc_secure=http_secure,
        auth_credentials=auth,
        headers={},
    )


def get_rerank_score(obj: Any) -> float | None:
    meta = getattr(obj, "metadata", None)
    if meta is None:
        return None
    rerank_score = getattr(meta, "rerank_score", None)
    if rerank_score is not None:
        return float(rerank_score)
    score = getattr(meta, "score", None)
    return float(score) if score is not None else None

