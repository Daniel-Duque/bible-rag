# Developer shortcuts for the Python RAG service. Requires uv (https://docs.astral.sh/uv/).
.PHONY: setup ingest serve chat test test-fast lint format

setup:      ## create .venv and install runtime + dev dependencies
	uv sync

ingest:     ## parse the PDFs and build the FAISS indexes (downloads the embedding model once)
	uv run bible-rag ingest

serve:      ## HTTP service on 127.0.0.1:8000 (models stay loaded)
	uv run bible-rag serve

chat:       ## interactive terminal session
	uv run bible-rag chat

test:       ## full test suite (~1 min; parses the real PDFs, builds tiny local models)
	uv run pytest

test-fast:  ## unit tests only (~15 s)
	uv run pytest -m "not pdf and not hf"

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .
