# Embeddings pipeline (MiniLM + FAISS)

## Setup

Important: Use Python 3.10 or 3.11 (PyTorch wheels are not yet available for 3.13 on macOS at the time of writing).

```bash
# 1) Create a fresh venv with Python 3.10/3.11
python3 -V  # ensure 3.10/3.11; if not, install via pyenv/homebrew
python3 -m venv .venv && source .venv/bin/activate

# 2) Install PyTorch (CPU-only)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) Install remaining deps
pip install -r embeddings/requirements.txt

# 4) Optional env file
cp embeddings/env.example .env || true
```

## Ingest CSV (`embeddings/data.csv`)

```bash
# One-time full ingest from CSV
python embeddings/ingest_csv.py --csv ./embeddings/data.csv --out ./embeddings/store

# Incremental re-index (adds new, removes missing)
python embeddings/ingest_csv.py --csv ./embeddings/data.csv --out ./embeddings/store --incremental

# Watch the CSV and re-index on change (debounced)
python embeddings/watch_csv.py --csv ./embeddings/data.csv --out ./embeddings/store
```

- Configure CSV/text columns, keys and watcher in `embeddings/config.yaml`

## Search

```bash
# Python FastAPI service (served by docker-compose on :8000)
curl 'http://localhost:8000/api/search?q=buscar%20producto%20rojo&k=10'

# Or CLI
python embeddings/search.py --index ./embeddings/store --k 10 --query "buscar producto rojo"
```

## Notes
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- Vectors are normalized; FAISS `IndexFlatIP` approximates cosine similarity
- Metadata is stored in `meta.sqlite`; vectors in `index.faiss`
- CSV selection: by default uses textual columns like `_source.file_name_text`, `_source.product_name_text`, `_source.project_name_text`, `_source.asset_type_name`, `_source.customer` if present; override in config.

## Troubleshooting
- Error "No matching distribution found for torch": your Python is likely 3.12/3.13; recreate venv with 3.10/3.11 and install torch as above.
- Error "ModuleNotFoundError: No module named 'numpy'": run `pip install -r embeddings/requirements.txt` in your activated venv.

