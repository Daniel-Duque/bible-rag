"""Web chat for the Bible RAG: `uv run streamlit run app.py` (see README to serve it online)."""

import streamlit as st

from bible_rag.config import Settings
from bible_rag.index import IndexNotFoundError, build_indexes
from bible_rag.rag import BibleRAG
from bible_rag.sources import VERSIONS

st.set_page_config(page_title="Bible RAG · KJV / RVR1960", page_icon="📖")


@st.cache_resource(show_spinner=False)
def get_rag() -> BibleRAG:
    """Loaded once per server and shared by all sessions."""
    settings = Settings.from_env()
    try:
        return BibleRAG.from_settings(settings)
    except IndexNotFoundError:  # fresh server: build the indexes once
        build_indexes(settings)
        return BibleRAG.from_settings(settings)


def show_details(item: dict) -> None:
    if item["unverified"]:
        st.warning(
            "Not found in the retrieved passages, verify before trusting: "
            + "; ".join(f"“{u}”" for u in item["unverified"])
        )
    # open the evidence when the answer could not be verified
    with st.expander(f"Sources · answered by {item['model']}", expanded=not item["grounded"]):
        for citation, text in item["sources"]:
            st.markdown(f"**{citation}**  \n{text}")
        for note in item["notes"]:
            st.caption(note)


with st.spinner("Loading models… (a first start also builds the search index, a few minutes)"):
    rag = get_rag()

labels = {VERSIONS[key].label: key for key in rag.indexes}
with st.sidebar:
    chosen = st.multiselect("Translations", list(labels), default=list(labels))
    k = st.slider("Passages per translation", 1, 5, rag.settings.top_k)
    st.caption(f"Search: `{rag.embedder_spec}` · Answers: `{rag.chain.active}`")
    for note in rag.fallback_notes():
        st.caption(note)
    if st.button("Clear chat"):
        st.session_state.history = []
    st.divider()
    for key in rag.indexes:  # texts and their licenses
        v = VERSIONS[key]
        st.caption(f"**{v.label}**: {v.name}. {v.license}. {v.conditions}")
        if not v.public:
            st.warning(f"{v.label} may not be served publicly.")

st.title("📖 Bible RAG")
st.caption(
    "Ask in English or Spanish · Pregunte en inglés o español. "
    "Answers cite the King James Version and the Reina-Valera 1960."
)

history = st.session_state.setdefault("history", [])
for item in history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant":
            show_details(item)

if question := st.chat_input("What does the Bible say about…? / ¿Qué dice la Biblia sobre…?"):
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer = rag.ask_stream(question, k, [labels[c] for c in chosen] or None)
        st.write_stream(answer)
        item = {
            "role": "assistant",
            "content": answer.text,
            "model": answer.model,
            "notes": answer.notes,
            "grounded": answer.grounding.grounded,
            "unverified": [c.text for c in answer.grounding.unsupported],
            "sources": [(h.passage.citation, h.passage.text) for h in answer.sources],
        }
        show_details(item)
    history.append(item)
