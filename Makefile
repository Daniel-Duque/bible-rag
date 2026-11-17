# Shortcuts for the common commands. Requires uv (https://docs.astral.sh/uv/).
.PHONY: setup app chat serve ingest sample prebuild test test-fast lint format

setup:      ## create .venv and install the dependencies
	uv sync

app:        ## web chat on http://localhost:8501
	uv run streamlit run app.py

chat:       ## interactive terminal session
	uv run bible-rag chat

serve:      ## read-only API on 127.0.0.1:8000 (see README: GPU machine, BIBLE_RAG_API_KEY)
	uv run bible-rag serve

ingest:     ## update the local indexes if a text or the code reading it changed (else instant)
	uv run bible-rag ingest

sample:     ## index ~1/4 of the Bible only (quick test of a new embedder)
	uv run bible-rag ingest --sample

prebuild:   ## regenerate the shipped indexes in storage/prebuilt/ (maintainers)
	BIBLE_RAG_VERSIONS=all uv run bible-rag prebuild

test:       ## full suite (~4 min: real Bible texts, the RVR1960 PDF if present, tiny models)
	uv run pytest

test-fast:  ## unit tests only, offline (~1 min)
	uv run pytest -m "not pdf and not hf and not sword"

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .
