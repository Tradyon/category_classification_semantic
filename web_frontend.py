import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from weaviate_search import (
    adaptive_hs_top_k,
    fuzzy_hs_match,
    fuzzy_hs_thresholds,
    hybrid_search_best_attribute,
    hybrid_search_top1,
    hs6_from_country_match,
    keep_hs_if_confident,
    keep_if_score_at_least,
)
from weaviate_utils import make_weaviate_client


INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>HS + Attributes Search</title>
    <style>
      :root { color-scheme: light dark; }
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; }
      .wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
      h1 { margin: 0 0 12px; font-size: 20px; }
      form { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
      input[type="text"] { flex: 1; min-width: 260px; padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(127,127,127,.4); }
      button { padding: 10px 14px; border-radius: 10px; border: 1px solid rgba(127,127,127,.4); cursor: pointer; }
      .opts { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 10px; opacity: .9; }
      .opts label { display: inline-flex; gap: 8px; align-items: center; font-size: 13px; }
      .opts input { width: 90px; padding: 6px 8px; border-radius: 8px; border: 1px solid rgba(127,127,127,.4); }
      .status { margin-top: 12px; font-size: 13px; opacity: .85; }
      .grid { margin-top: 16px; display: grid; grid-template-columns: 1fr; gap: 12px; }
      .card { border: 1px solid rgba(127,127,127,.3); border-radius: 14px; padding: 14px; }
      .k { font-size: 12px; opacity: .75; margin-bottom: 6px; }
      .v { font-size: 14px; white-space: pre-wrap; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .row { display: flex; gap: 10px; flex-wrap: wrap; }
      .pill { display: inline-block; padding: 3px 8px; border: 1px solid rgba(127,127,127,.35); border-radius: 999px; font-size: 12px; opacity: .9; }
      .err { color: #b00020; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>HS Code + Country HS + Attributes Search</h1>
      <form id="f">
        <input id="q" type="text" placeholder="Search (e.g. organic green coffee, avocado, black pepper)" autocomplete="off" />
        <button type="submit">Search</button>
      </form>
	      <div class="opts">
	        <label>min hs score <input id="minHs" type="number" step="0.1" value="0.0" /></label>
	        <label>min attribute score <input id="minAttr" type="number" step="0.1" value="0.0" /></label>
	      </div>
      <div id="status" class="status"></div>
      <div class="grid">
        <div class="card">
          <div class="k">6-digit HS</div>
          <div id="hs6" class="v mono"></div>
        </div>
        <div class="card">
          <div class="k">Country-specific HS (8–12 digit)</div>
          <div id="hsc" class="v mono"></div>
        </div>
        <div class="card">
          <div class="k">Attribute triple</div>
          <div id="attr" class="v mono"></div>
        </div>
      </div>
      <div class="status">Tip: If results are empty, lower the thresholds.</div>
    </div>
    <script>
      const $ = (id) => document.getElementById(id);
      const fmt = (obj) => obj ? JSON.stringify(obj, null, 2) : "(no confident match)";

	      async function runSearch(q) {
        $("status").textContent = "Searching…";
        $("hs6").textContent = "";
        $("hsc").textContent = "";
        $("attr").textContent = "";
	        const minHs = parseFloat($("minHs").value || "0");
	        const minAttr = parseFloat($("minAttr").value || "0");
	        const url = `/api/search?q=${encodeURIComponent(q)}&min_hs6_score=${encodeURIComponent(minHs)}&min_hs_country_score=${encodeURIComponent(minHs)}&min_attribute_score=${encodeURIComponent(minAttr)}`;
	        const res = await fetch(url);
        const data = await res.json();
        if (!res.ok) {
          $("status").innerHTML = `<span class="err">Error:</span> ${data.error || res.status}`;
          return;
        }
        $("status").textContent = `OK`;
        $("hs6").textContent = fmt(data.hs6);
        $("hsc").textContent = fmt(data.hs_country);
        $("attr").textContent = fmt(data.attribute);
      }

      $("f").addEventListener("submit", (e) => {
        e.preventDefault();
        const q = ($("q").value || "").trim();
        if (!q) return;
        runSearch(q);
      });

      // default query
      $("q").value = "organic green coffee";
      runSearch($("q").value);
    </script>
  </body>
</html>
"""


def build_response(
    query: str,
    *,
    alpha: float,
    top_k: int,
    rerank_top_n: int,
    min_hs6_score: float,
    min_hs_country_score: float,
    min_attribute_score: float,
    fuzzy_hs: bool,
    fuzzy_hs_min_avg: float,
    fuzzy_hs_min_max: float,
    weaviate_url: str | None,
    weaviate_api_key: str | None,
    grpc_port: int | None,
) -> dict[str, Any]:
    q = query.strip()
    if not q:
        return {"query": query, "hs6": None, "hs_country": None, "attribute": None}

    client = make_weaviate_client(
        weaviate_url=weaviate_url, weaviate_api_key=weaviate_api_key, grpc_port=grpc_port
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

        hs_top_k = adaptive_hs_top_k(q, top_k)
        hs_rerank_top_n = max(rerank_top_n, hs_top_k)
        fuzzy_avg, fuzzy_max = fuzzy_hs_thresholds(q, fuzzy_hs_min_avg, fuzzy_hs_min_max)

        hs_country_raw = hybrid_search_top1(
            client,
            collection_name="HSCountry",
            query=q,
            alpha=alpha,
            top_k=hs_top_k,
            rerank_top_n=hs_rerank_top_n,
            query_properties=hs_query_props["HSCountry"],
        )
        hs_country = keep_hs_if_confident(hs_country_raw, min_hs_country_score, query=q)
        if hs_country is None and fuzzy_hs:
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
                alpha=alpha,
                top_k=hs_top_k,
                rerank_top_n=hs_rerank_top_n,
                query_properties=hs_query_props["HS6"],
            )
        hs6 = keep_hs_if_confident(hs6, min_hs6_score, query=q)
        if hs6 is None and fuzzy_hs:
            hs6 = fuzzy_hs_match(
                client,
                collection_name="HS6",
                query=q,
                min_avg=fuzzy_avg,
                min_max=fuzzy_max,
            )

        attr = hybrid_search_best_attribute(
            client,
            query=q,
            alpha=alpha,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            min_score=min_attribute_score,
            query_properties=hs_query_props["Attributes"],
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
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "HSFrontend/0.1"

    def _send(self, status: int, body: bytes, *, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if parsed.path == "/api/search":
            qs = parse_qs(parsed.query)
            q = (qs.get("q", [""])[0] or "").strip()

            def fparam(name: str, default: float) -> float:
                raw = (qs.get(name, [None])[0] or "").strip()
                if not raw:
                    return default
                try:
                    return float(raw)
                except Exception:  # noqa: BLE001
                    return default

            alpha = fparam("alpha", self.server.cfg["alpha"])
            min_hs6_score = fparam("min_hs6_score", self.server.cfg["min_hs6_score"])
            min_hs_country_score = fparam("min_hs_country_score", self.server.cfg["min_hs_country_score"])
            min_attribute_score = fparam("min_attribute_score", self.server.cfg["min_attribute_score"])

            def iparam(name: str, default: int) -> int:
                raw = (qs.get(name, [None])[0] or "").strip()
                if not raw:
                    return default
                try:
                    return int(raw)
                except Exception:  # noqa: BLE001
                    return default

            top_k = iparam("top_k", self.server.cfg["top_k"])
            rerank_top_n = iparam("rerank_top_n", self.server.cfg["rerank_top_n"])

            try:
                data = build_response(
                    q,
                    alpha=alpha,
                    top_k=top_k,
                    rerank_top_n=rerank_top_n,
                    min_hs6_score=min_hs6_score,
                    min_hs_country_score=min_hs_country_score,
                    min_attribute_score=min_attribute_score,
                    fuzzy_hs=self.server.cfg["fuzzy_hs"],
                    fuzzy_hs_min_avg=self.server.cfg["fuzzy_hs_min_avg"],
                    fuzzy_hs_min_max=self.server.cfg["fuzzy_hs_min_max"],
                    weaviate_url=self.server.cfg.get("weaviate_url"),
                    weaviate_api_key=self.server.cfg.get("weaviate_api_key"),
                    grpc_port=self.server.cfg.get("grpc_port"),
                )
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.OK, body, content_type="application/json; charset=utf-8")
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, body, content_type="application/json; charset=utf-8")
            return

        self._send(HTTPStatus.NOT_FOUND, b"Not Found", content_type="text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> int:
    p = argparse.ArgumentParser(description="Tiny local frontend for Weaviate HS/attrs search.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--weaviate-url", type=str, default=None)
    p.add_argument("--weaviate-api-key", type=str, default=None)
    p.add_argument("--grpc-port", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rerank-top-n", type=int, default=5)
    p.add_argument("--min-hs-score", type=float, default=0.0)
    p.add_argument("--min-attribute-score", type=float, default=0.0)
    p.add_argument("--fuzzy-hs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fuzzy-hs-min-avg", type=float, default=0.84)
    p.add_argument("--fuzzy-hs-min-max", type=float, default=0.9)
    args = p.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.cfg = {
        "weaviate_url": args.weaviate_url,
        "weaviate_api_key": args.weaviate_api_key,
        "grpc_port": args.grpc_port,
        "alpha": args.alpha,
        "top_k": args.top_k,
        "rerank_top_n": args.rerank_top_n,
        "min_hs6_score": args.min_hs_score,
        "min_hs_country_score": args.min_hs_score,
        "min_attribute_score": args.min_attribute_score,
        "fuzzy_hs": args.fuzzy_hs,
        "fuzzy_hs_min_avg": args.fuzzy_hs_min_avg,
        "fuzzy_hs_min_max": args.fuzzy_hs_min_max,
    }

    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
