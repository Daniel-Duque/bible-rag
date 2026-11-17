"""Decisions by meaning instead of keywords, with the multilingual embedder: what a question
asks (routing), what it is about (topic), and which way a text leans (opposition). Examples
are given in English and Spanish, but the embedder places other languages in the same space,
so "Existem paradoxos na Bíblia sobre o álcool?" is routed and matched like its English form.

Opposition: pairs of approving / condemning statements ("Blessed is the man that…" / "Woe
unto him that…", in several languages) define a direction in the embedding space, the
*opposition axis*. A text's position on it says how approving or condemning it is, whatever its
words; moving a topic to either end of it gives its positive and negative form (used when no
LLM can write the two statements).

The embeddings of these fixed texts are computed once per embedder and cached
(storage/index/semantics-<embedder>.npz; shipped in storage/prebuilt/).
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from pathlib import Path

import numpy as np

from .lexicon import TOPICS, Topic

log = logging.getLogger(__name__)

GENERAL, CONTRADICTION, TRANSLATION = "general", "contradiction", "translation"
INTENTS = (GENERAL, CONTRADICTION, TRANSLATION)

# Example questions per intent, on varied subjects so that the subject doesn't decide
INTENT_EXAMPLES = {
    GENERAL: [
        "What does the Bible say about forgiveness?",
        "Who was Moses?",
        "Why did Jesus weep?",
        "How many days did it rain during the flood?",
        "What is the greatest commandment?",
        "Who wrote the book of Romans?",
        "What happened at Pentecost?",
        "How should I pray?",
        "What does Psalm 23 mean?",
        "Tell me about the prophet Elijah",
        "What is grace?",
        "Is it a sin to lie?",
        "How do I settle an argument with my neighbour?",
        "What does the Bible teach about drinking?",
        "Quote the verse about love being patient",
        "Is Jesus the Son of God?",
        "Does God hear our prayers?",
        "Is heaven real?",
        "¿Qué dice la Biblia sobre la humildad?",
        "¿Quién fue el rey David?",
        "¿Por qué lloró Jesús?",
        "¿Qué pasó en el mar Rojo?",
        "¿Cómo debo orar?",
        "¿Quién era Pablo?",
        "¿Qué es la fe?",
        "¿Es pecado enojarse?",
        "¿Cómo resolver una pelea entre hermanos?",
        "¿Jesús es el Hijo de Dios?",
        "¿Dios escucha nuestras oraciones?",
        "¿Existe el cielo?",
    ],
    CONTRADICTION: [
        "Does the Bible contradict itself about violence?",
        "Is there a paradox in what the Bible says about riches?",
        "Are there inconsistencies about the law of Moses?",
        "Some verses allow it and others forbid it: which is right?",
        "Do Paul and James disagree about salvation?",
        "The Bible says both yes and no about oaths, how so?",
        "Is gambling good or bad according to different verses?",
        "Is gossip permitted or prohibited by Scripture?",
        "Are there opposite teachings about anger?",
        "Which verses support it and which verses condemn it?",
        "Isn't it contradictory that God rests and never sleeps?",
        "Some passages praise it while others condemn it. Why?",
        "One verse says one thing and another verse says the opposite",
        "¿Se contradice la Biblia sobre la venganza?",
        "¿Hay paradojas sobre la riqueza en la Biblia?",
        "¿Unos versículos lo permiten y otros lo prohíben?",
        "¿Hay enseñanzas opuestas sobre la ira?",
        "¿Pablo y Santiago se contradicen?",
        "¿Es bueno o malo según la Biblia? Hay versículos a favor y en contra",
        "Un versículo dice una cosa y otro dice lo contrario",
    ],
    TRANSLATION: [
        "How is this verse translated in other versions?",
        "Does the King James say the same as the Reina-Valera?",
        "Which translation is more accurate for Isaiah 7:14?",
        "Why do Bible versions use different words here?",
        "What is the original Greek word behind this translation?",
        "Compare the wording of John 1:1 in each version",
        "Does the meaning change between translations?",
        "How do the English and Spanish Bibles render 'grace'?",
        "Is this word translated differently in the Spanish Bible?",
        "Show me Romans 8:28 in every version",
        "¿Cómo se traduce este versículo en otras versiones?",
        "¿Dice lo mismo la Reina-Valera que la King James?",
        "¿Por qué las versiones usan palabras distintas aquí?",
        "Compara la redacción de Juan 3:16 en cada traducción",
        "¿Cambia el significado entre traducciones?",
        "¿Cuál es la palabra original en hebreo?",
    ],
}

# Bible subjects outside the lexicon: a question nearer to these than to any lexicon topic is
# about a topic of its own (searched with its own words)
OTHER_TOPICS = [
    "prayer",
    "baptism",
    "angels",
    "tithing and offerings",
    "dancing",
    "tattoos",
    "music and singing",
    "women in the church",
    "the resurrection of Jesus",
    "the creation of the world",
    "the end times",
    "miracles and healing",
    "fasting",
    "marriage",
    "children and parents",
    "work and laziness",
    "the death penalty",
    "the Holy Spirit",
    "heaven",
    "demons",
    "predestination",
    "the Trinity",
    "long hair",
    "the birth of Jesus",
    "animals",
    "prophecy",
    "circumcision",
    "idols",
    "wisdom",
    "sickness",
    "la oración",
    "el bautismo",
    "los ángeles",
    "el diezmo",
    "el baile",
    "los tatuajes",
    "la música",
    "la mujer en la iglesia",
    "la resurrección",
    "la creación",
    "el fin del mundo",
    "los milagros",
    "el ayuno",
    "el matrimonio",
    "los hijos",
    "el trabajo",
    "el cielo",
    "los demonios",
    "la predestinación",
    "la Trinidad",
    "los ídolos",
    "la sabiduría",
]

_NULL_TEMPLATES = (
    "What does the Bible say about {}?",
    "Are there contradictions in the Bible about {}?",
)

# (approving, opposing) statements: their differences, averaged, are the opposition axis. Most
# are in the Bible's own register ("Woe unto him that…", "an abomination unto the LORD"),
# since that is how the passages express approval and condemnation.
OPPOSITION_PAIRS = [
    ("Blessed is the man that doeth this.", "Woe unto him that doeth this."),
    ("This is good in the sight of the LORD.", "This is an abomination unto the LORD."),
    ("It maketh glad the heart of man.", "It is a mocker, and bringeth sorrow."),
    ("Do it, and it shall be well with thee.", "Thou shalt not do it, lest thou die."),
    ("It is a gift of God; rejoice therein.", "It is sin; beware of it."),
    (
        "They that do it shall inherit the kingdom.",
        "They that do it shall not inherit the kingdom.",
    ),
    ("Praise the LORD with it.", "Be not among them that do it."),
    ("It is lawful and clean.", "It is unlawful and unclean."),
    ("The LORD commanded it.", "The LORD forbade it."),
    ("Bienaventurado el varón que lo hace.", "¡Ay del que lo hace!"),
    ("Es bueno delante de Jehová.", "Es abominación a Jehová."),
    ("Alegra el corazón del hombre.", "Es escarnecedor, y trae dolor."),
    ("Hazlo, y te irá bien.", "No lo hagas, no sea que mueras."),
    ("Es don de Dios; gozaos en ello.", "Es pecado; guardaos de ello."),
    ("Es lícito y limpio.", "Es ilícito e inmundo."),
    ("Los que lo hacen heredarán el reino.", "Los que lo hacen no heredarán el reino de Dios."),
    ("This is good and allowed.", "This is evil and forbidden."),
    ("Esto es bueno y permitido.", "Esto es malo y prohibido."),
    ("Isto é bom e permitido.", "Isto é mau e proibido."),
    ("C'est bon et permis.", "C'est mauvais et interdit."),
    ("Das ist gut und erlaubt.", "Das ist böse und verboten."),
]


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def topic_examples(topic: Topic) -> list[str]:
    """What a topic is about, in words to compare questions with."""
    texts = [*topic.examples, topic.name["en"], topic.name["es"]]
    texts += [c.gloss for c in topic.concepts]
    if topic.claims:
        texts += [*topic.claims["en"], *topic.claims["es"]]
    return list(dict.fromkeys(texts))


class Semantics:
    """The embedded examples, computed once per embedder (thread-safe, lazily)."""

    top = 3  # an intent's score: mean similarity of its `top` closest examples

    def __init__(self, embedder, cache: Path | None = None, shipped: Path | None = None):
        self.embedder = embedder
        self._lock = threading.RLock()  # properties build on each other
        self._intents: dict[str, np.ndarray] | None = None
        self._topics: list[tuple[Topic | None, np.ndarray]] | None = None
        self._axis: np.ndarray | None = None
        self._threshold: float | None = None
        # embeddings of the fixed texts (examples, topics, pairs), kept on disk: a new process
        # computes none of them; changing an example re-embeds only that one
        self.cache, self._known = cache, {}
        for path in (shipped, cache):
            if path is not None and path.exists():
                data = np.load(path)
                if str(data["model"]) == embedder.model_name:
                    texts = data["texts"].tobytes().decode().split("\x00")
                    self._known.update(zip(texts, data["vectors"].astype(np.float32), strict=True))

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(texts)

    def fixed(self, texts: list[str]) -> np.ndarray:
        """Embeddings of fixed texts, from the cache when known."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._known]
        if missing:
            self._known.update(zip(missing, self.embed(missing), strict=True))
            self._save()
        return np.array([self._known[t] for t in texts], np.float32)

    def _save(self) -> None:
        if self.cache is None:
            return
        texts = list(self._known)
        try:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            from .index import save_npz

            save_npz(
                self.cache,
                True,
                texts=np.frombuffer("\x00".join(texts).encode(), np.uint8),
                vectors=np.array([self._known[t] for t in texts], np.float16),
                model=np.array(self.embedder.model_name),
            )
        except OSError as e:  # read-only: fine, recomputed next time
            log.debug("Could not save %s: %s", self.cache, e)

    # ------------------------------------------------------------------------ routing
    @property
    def intents(self) -> dict[str, np.ndarray]:
        with self._lock:
            if self._intents is None:
                self._intents = {k: self.fixed(v) for k, v in INTENT_EXAMPLES.items()}
        return self._intents

    def intent_scores(self, qvec: np.ndarray) -> dict[str, float]:
        return {k: float(np.sort(v @ qvec)[-self.top :].mean()) for k, v in self.intents.items()}

    def route(self, qvec: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """(intent, margin over the runner-up, scores)."""
        scores = self.intent_scores(qvec)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[0], scores[ranked[0]] - scores[ranked[1]], scores

    def without_intent(self, qvec: np.ndarray, intent: str) -> np.ndarray:
        """The question with its task ("are there contradictions…") projected out: what it
        is about."""
        c = unit(self.intents[intent].mean(axis=0))
        return unit(qvec - float(qvec @ c) * c)

    # ------------------------------------------------------------------------- topics
    @property
    def topics(self) -> list[tuple[Topic | None, np.ndarray]]:
        with self._lock:
            if self._topics is None:
                self._topics = [(t, self.fixed(topic_examples(t))) for t in TOPICS]
                self._topics.append((None, self.fixed(OTHER_TOPICS)))
        return self._topics

    @property
    def topic_threshold(self) -> float:
        """How close a question must be to a lexicon topic to be about it: the 90th
        percentile of the closeness of questions about *other* subjects ("What does the Bible
        say about prayer?"), computed for this embedder, whose similarity scale is its own."""
        with self._lock:
            if self._threshold is None:
                questions = [t.format(o) for o in OTHER_TOPICS for t in _NULL_TEMPLATES]
                scores = [self._best(self.subject(v))[1] for v in self.fixed(questions)]
                self._threshold = float(np.quantile(scores, 0.9))
        return self._threshold

    def _best(self, vec: np.ndarray) -> tuple[Topic | None, float]:
        scored = [(float((v @ vec).max()), t) for t, v in self.topics if t is not None]
        score, topic = max(scored, key=lambda s: s[0])
        return topic, score

    def subject(self, qvec: np.ndarray) -> np.ndarray:
        """What a question is about: the question minus its task (routing's intent)."""
        return self.without_intent(qvec, self.route(qvec)[0])

    def topic(self, vec: np.ndarray) -> tuple[Topic | None, float]:
        """The lexicon topic closest in meaning to a subject vector (see `subject`), or None
        when the subject is none of them (closer to another subject, or not close enough)."""
        other = float((self.topics[-1][1] @ vec).max())
        topic, score = self._best(vec)
        if score < other or score < self.topic_threshold:
            return None, score
        return topic, score

    def subject_words(self, question: str, qvec: np.ndarray) -> list[str]:
        """The words of a question that carry its subject, not its task, found by leaving
        each word out: a subject word moves the subject when removed, a task word
        ("paradoxos", "contradict", "Bible") lowers the intent instead. Any language."""
        words = list(dict.fromkeys(w for w in re.findall(r"\w+", question) if len(w) > 2))
        if not words:
            return []
        intent, _, scores = self.route(qvec)
        subject = self.without_intent(qvec, intent)
        variants = [re.sub(rf"\b{re.escape(w)}\b", " ", question) for w in words]
        found = []
        for w, v in zip(words, self.embed(variants), strict=True):
            task = scores[intent] - self.intent_scores(v)[intent]  # intent lost without w
            moved = 1 - float(self.without_intent(v, intent) @ subject)  # subject moved
            found.append((w, task, moved))
        most_task = max(t for _, t, _ in found)
        most_moved = max(m for _, _, m in found)
        return [
            w.lower()
            for w, task, moved in found
            if task < 0.7 * most_task and task < 0.25 * moved and moved >= 0.35 * most_moved
        ]

    # --------------------------------------------------------------------- opposition
    @property
    def axis(self) -> np.ndarray:
        """Direction from opposing to approving statements."""
        with self._lock:
            if self._axis is None:
                pos = self.fixed([p for p, _ in OPPOSITION_PAIRS])
                neg = self.fixed([n for _, n in OPPOSITION_PAIRS])
                self._axis = unit(np.mean([unit(d) for d in pos - neg], axis=0))
        return self._axis

    def approval(self, vectors: np.ndarray) -> np.ndarray:
        """How approving (> 0) or opposing (< 0) each text is."""
        return vectors @ self.axis

    def polarize(self, vec: np.ndarray, strength: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """A topic's positive and negative form: the topic moved to either end of the axis
        ("dancing" -> ~"dancing is approved" / ~"dancing is condemned")."""
        base = vec - float(vec @ self.axis) * self.axis  # neutral: no approval either way
        return unit(base + strength * self.axis), unit(base - strength * self.axis)

    def opposed(self, a: np.ndarray, b: np.ndarray) -> float:
        """How much statement `a` approves more than `b` (> 0: a is the positive one)."""
        return float((a - b) @ self.axis)


_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.Lock()


def semantics_for(embedder, cache: Path | None = None, shipped: Path | None = None) -> Semantics:
    """One Semantics per embedder (its example embeddings are computed once, and kept in
    `cache`; `shipped`: the prebuilt copy)."""
    with _CACHE_LOCK:
        if embedder not in _CACHE:
            _CACHE[embedder] = Semantics(embedder, cache, shipped)
        return _CACHE[embedder]
