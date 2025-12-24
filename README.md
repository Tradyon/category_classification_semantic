# Category Classification Semantic — quick guide

## Setup
- Copy `.env.example` to `.env` and fill in keys as needed (Clarifai for OCR, Weaviate URL/API key if not local).
- Install deps: `uv sync` (or `pip install -e .`).

## Extract HS codes
- Gemini direct: (need not redo, just use the cache)  
  `uv run extract_hs_codes_gemini.py --input "HS Code" --pages-jsonl outputs/hs_pages_gemini.jsonl --rows-csv outputs/hs_rows_gemini.csv`  
  Uses Gemini structured output, inherits parent headings for child rows.

## Index into Weaviate (hybrid + rerank)
1) Start local Weaviate modules: `docker compose up -d`  
To index everything using cached augmentations (if available, just run this and you can skip to search):  
`uv run weaviate_index.py --hs-csv outputs/hs_rows_gemini.csv --attrs-json attrs.json --recreate --llm-augment --llm-mode keywords --llm-threshold 80 --llm-augment-attrs`

2) Index HS + attributes:  
   `uv run weaviate_index.py --hs-csv outputs/hs_rows_gemini.csv --attrs-json attrs.json --recreate`
Models used in `docker-compose.yml`: embeddings via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, reranking via `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
Optional: augment HS descriptions + add multilingual synonyms via Gemini before indexing (recommended for better recall):  
`uv run weaviate_index.py --recreate --llm-augment --llm-mode keywords --llm-threshold 80`  
If you want concise canonical descriptions for *every* HS row (not just short/generic ones), use `--llm-mode all`.  
To refresh previously cached augmentations, add `--llm-force`. To speed up LLM calls, set `--llm-workers 4` (or higher if your quota allows).
Optional: also add multilingual keywords for `attrs.json` triples (stored in `.cache/attrs_augment/augmented_attributes.jsonl`):  
`uv run weaviate_index.py --recreate --llm-augment-attrs`  
This also caches product/type synonyms in `.cache/attrs_augment/augmented_products.jsonl` and `.cache/attrs_augment/augmented_attribute_types.jsonl`.
Caches are written as “latest per id”; when a cache file changes, the previous version is stashed under `.cache/_history/...`.

## Search (CLI)
Run hybrid + rerank across 6-digit HS, country HS (8–12 digit), and attribute triples:  
`uv run weaviate_search.py "organic green coffee"`  
Flags: `--alpha` (hybrid mix), `--top-k`, `--rerank-top-n`, and score cutoffs `--min-hs6-score`, `--min-hs-country-score`, `--min-attribute-score`. The HS6 result will follow the top country-specific prefix when available.

## Frontend (port 8000)
Start a minimal UI:  
`uv run web_frontend.py --host 127.0.0.1 --port 8000`  
Open `http://localhost:8000`, search, and adjust score thresholds in the page.
