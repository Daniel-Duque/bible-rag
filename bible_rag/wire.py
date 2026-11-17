"""What the API sends: plans, passages and answers as plain JSON (and back), so that an answer
from the GPU server looks to the CLI and the web app exactly like a local one."""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Passage
from .grounding import Check, Grounding
from .index import Hit

PROTOCOL = 3  # bumped on incompatible changes; client and server must agree


def hit_to_json(hit: Hit) -> dict:
    p = hit.passage
    return {
        "version": p.version,
        "language": p.language,
        "book": p.book,
        "chapter": p.chapter,
        "verse_start": p.verse_start,
        "verse_end": p.verse_end,
        "text": p.text,
        "score": round(float(hit.score), 4),
    }


def hit_from_json(data: dict) -> Hit:
    passage = Passage(
        str(data["version"]),
        str(data["language"]),
        str(data["book"]),
        int(data["chapter"]),
        int(data["verse_start"]),
        int(data["verse_end"]),
        str(data["text"]),
    )
    return Hit(passage, float(data.get("score", 0.0)))


def plan_to_json(plan) -> dict:
    return {
        "intent": plan.intent,
        "language": plan.language,
        "reply_language": plan.reply_language,
        "topic_name": plan.topic_name,
        "reason": plan.reason,
        "queries": list(plan.queries),
        "cited": [hit_to_json(h) for h in plan.extra],
    }


@dataclass
class RemotePlan:
    """The parts of an agent.Plan the CLI and the web app show."""

    intent: str
    language: str
    reply_language: str = ""
    topic_name: str = ""
    reason: str = ""
    queries: list[str] = field(default_factory=list)
    extra: list[Hit] = field(default_factory=list)  # passages cited by notes

    @classmethod
    def from_json(cls, data: dict) -> RemotePlan:
        return cls(
            str(data["intent"]),
            str(data["language"]),
            str(data.get("reply_language", "")),
            str(data.get("topic_name", "")),
            str(data.get("reason", "")),
            [str(q) for q in data.get("queries", [])],
            [hit_from_json(h) for h in data.get("cited", [])],
        )


def grounding_to_json(g: Grounding) -> dict:
    def checks(items):
        return [{"text": c.text, "supported": c.supported} for c in items]

    return {"references": checks(g.references), "quotes": checks(g.quotes)}


def grounding_from_json(data: dict) -> Grounding:
    def checks(items):
        return [Check(str(c["text"]), bool(c["supported"])) for c in items]

    return Grounding(checks(data.get("references", [])), checks(data.get("quotes", [])))


def answer_to_json(answer) -> dict:
    return {
        "text": answer.text,
        "synthesis": answer.synthesis,
        "model": answer.model,
        "notes": list(answer.notes),
        "grounding": grounding_to_json(answer.grounding),
        "sources": [hit_to_json(h) for h in answer.sources],
    }
