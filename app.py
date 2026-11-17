"""Web chat: `uv run streamlit run app.py`, then open http://localhost:8501.

The assistant reads each question and chooses how to answer it: a direct search, both sides of
a contradiction, or a comparison of the translations (see bible_rag/agent.py). Buttons under
the last answer ask the same question with one of the other two tasks."""

import streamlit as st

from bible_rag.config import Settings
from bible_rag.index import IndexNotFoundError, build_indexes
from bible_rag.rag import BibleRAG
from bible_rag.remote import open_backend
from bible_rag.sources import VERSIONS

st.set_page_config(page_title="Bible RAG", page_icon="📖")

ROUTE_LABELS = {
    "general": "🔎 Direct search",
    "contradiction": "⚖️ Contradiction search: the two opposite statements were searched",
    "translation": "🌐 Translation comparison",
}
TASK_BUTTONS = {  # ask the same question with another task
    "contradiction": "⚖️ Look for contradictions",
    "translation": "🌐 Compare translations",
    "general": "🔎 Direct search",
}


@st.cache_resource(show_spinner=False)
def get_rag() -> BibleRAG:
    """Loaded once per server and shared by all sessions: the API server if configured
    (BIBLE_RAG_SERVER_URL) and reachable, else the local models."""
    settings = Settings.from_env()
    try:
        rag = open_backend(settings)
        if hasattr(rag, "warm_up"):  # local models: the one-time work before the first question
            rag.warm_up()
        return rag
    except IndexNotFoundError:  # no shipped index for this configuration: build it once
        status = st.status("Building the search index (a few minutes, only once)…")
        bars = {}

        def progress(version: str, done: int, total: int) -> None:
            if version not in bars:
                bars[version] = status.progress(0.0)
            bars[version].progress(done / total, text=f"{version}: {done} / {total} passages")

        build_indexes(settings, progress=progress)
        status.update(label="Search index built", state="complete")
        return open_backend(settings)


def route_caption(item: dict) -> str:
    """How the question was understood: the pipeline, the topic and why."""
    caption = ROUTE_LABELS[item["intent"]]
    caption += f" · {item['topic']}" if item.get("topic") else ""
    return caption + (f" · _{item['reason']}_" if item.get("reason") else "")


def show_details(item: dict) -> None:
    if item["unverified"]:
        st.warning(
            "Not found in the retrieved passages, verify before trusting: "
            + "; ".join(f"“{u}”" for u in item["unverified"])
        )
    # the evidence is opened when the answer could not be verified
    with st.expander(f"Sources · answered by {item['model']}", expanded=not item["grounded"]):
        if item["intent"] != "general":
            st.caption("Searched for: " + " | ".join(f"“{q}”" for q in item["queries"]))
        for citation, text in item["sources"]:
            st.markdown(f"**{citation}**  \n{text}")
        for note in item["notes"]:
            st.caption(note)


with st.spinner("Loading models…"):
    rag = get_rag()

labels = {VERSIONS[key].label: key for key in rag.indexes}
with st.sidebar:
    chosen = st.multiselect("Translations", list(labels), default=list(labels))
    k = st.slider("Passages per translation", 1, 5, rag.settings.top_k)
    st.caption(f"Search: `{rag.embedder_spec}` · Answers: `{rag.chain.describe()}`")
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
    "Answers cite every loaded translation: "
    + ", ".join(VERSIONS[key].name for key in rag.indexes)
    + ". Ask a direct question (“Where was Jesus born?”), about contradictions "
    "(“Are there paradoxes in the Bible about alcohol?”) or about translations "
    "(“How do the versions render Psalm 16:10?”)."
)


def task_buttons(item: dict, index: int) -> tuple[str, str] | None:
    """Buttons under an answer for the two other tasks: (question, task) if one was clicked."""
    others = [t for t in TASK_BUTTONS if t != item["intent"]]
    for column, task in zip(st.columns(len(others)), others, strict=True):
        if column.button(TASK_BUTTONS[task], key=f"task-{index}-{task}"):
            return item["question"], task
    return None


def answer(question: str, task: str | None) -> dict:
    """Show the question and stream its answer; returns the answer's history entry."""
    with st.chat_message("user"):
        if task:
            st.caption(f"Asked as: {TASK_BUTTONS[task]}")
        st.markdown(question)
    with st.chat_message("assistant"):
        result = rag.ask_stream(question, k, [labels[c] for c in chosen] or None, task)
        plan = result.plan
        st.caption(
            route_caption({"intent": plan.intent, "topic": plan.topic_name, "reason": plan.reason})
        )
        st.write_stream(result)
        item = {
            "role": "assistant",
            "question": question,
            "content": result.text,
            "intent": plan.intent,
            "topic": plan.topic_name,
            "reason": plan.reason,
            "queries": plan.queries,
            "model": result.model,
            "notes": result.notes,
            "grounded": result.grounding.grounded,
            "unverified": [c.text for c in result.grounding.unsupported],
            "sources": [(h.passage.citation, h.passage.text) for h in result.sources],
        }
        show_details(item)
    return item


history = st.session_state.setdefault("history", [])
requested = None  # (question, task) from a button under the last answer
for index, item in enumerate(history):
    with st.chat_message(item["role"]):
        if item.get("task"):
            st.caption(f"Asked as: {TASK_BUTTONS[item['task']]}")
        st.markdown(item["content"])
        if item["role"] == "assistant":
            st.caption(route_caption(item))
            show_details(item)
            if index == len(history) - 1:
                requested = task_buttons(item, index)

typed = st.chat_input("What does the Bible say about…? / ¿Qué dice la Biblia sobre…?")
if typed or requested:
    question, task = (typed, None) if typed else requested
    history.append({"role": "user", "content": question, "task": task})
    history.append(answer(question, task))
    st.rerun()  # redraw with the buttons under the new answer
