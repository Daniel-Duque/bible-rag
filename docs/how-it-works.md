# How it works

This page describes how the assistant turns a question into an answer, how the texts and
indexes are prepared, and how well it works. For installing and using it, see the
[README](../README.md).

- [Overview](#overview)
- [Routing: which kind of question](#routing-which-kind-of-question)
- [The three pipelines](#the-three-pipelines)
- [Contradictions: the two sides](#contradictions-the-two-sides)
- [Translations and the original languages](#translations-and-the-original-languages)
- [The lexicon](#the-lexicon)
- [Writing and checking the answer](#writing-and-checking-the-answer)
- [Texts and passages](#texts-and-passages)
- [Search indexes](#search-indexes)
- [Quality](#quality)

## Overview

```
question
  -> route by meaning: direct / contradiction / translation               semantic.py
  -> run the pipeline                                                     agent.py
       direct:         hybrid search in each translation
       contradiction:  topic -> its words in each translation + a search for the affirming
                       and the opposing statement -> stance of each passage -> two sides
       translation:    the verses cited, or the passages where the versions word a topic
                       differently
  -> the same verses in every other translation and in the Hebrew / Greek     align.py
  -> notes: wording differences, the original word (Strong's), known traps
  -> prompt for that kind of question -> first working LLM of the chain        rag.py
  -> synthesis + the passages per translation + citation check            grounding.py
```

Every step works without a language model. The LLM writes the synthesis and, when it is
capable enough, helps with two planning decisions (see below); without it the answer is the
lead sentence and the passages.

## Routing: which kind of question

Keyword lists break as soon as a question is phrased differently or asked in another language,
so the decisions are made with the multilingual embedder (`bible_rag/semantic.py`):

- **Kind of question.** The question is embedded and compared with 66 example questions in
  English and Spanish (30 direct, 20 contradiction, 16 translation), on varied subjects so that
  the subject doesn't decide. A kind's score is the mean similarity of its three closest
  examples. The embedder places other languages in the same space, so *"Existem paradoxos na
  Bíblia sobre o álcool?"* is routed like its English form.
- **Close calls.** When direct and another kind score within 0.03 of each other, a capable LLM
  (not the small fallback) decides, if one is loaded and `BIBLE_RAG_LLM_PLANNING` is on.
  Otherwise the direct search is used, which also shows every translation, and the answer says
  it was unsure ("unsure between translation and general").
- **Asking for a task.** Routing can be overridden: the web chat's buttons under the last
  answer, `bible-rag ask --task …`, the chat's `/contradiction` and `/translation` commands,
  and the API's `task` field run the chosen pipeline directly ("as requested").
- **Topic.** The question's task ("are there contradictions…") is projected out of its vector;
  what is left, its subject, is compared with the lexicon's topics and with 52 other Bible
  subjects. The threshold is calibrated per embedder from questions about those other
  subjects, so "jewelry" or "astrology" are recognised as subjects of their own, while
  "drinking beer" or "self-defence" are recognised as alcohol and war.
- **Subject words**, for a topic outside the lexicon, are found by leaving each word out of the
  question: removing a subject word moves the subject, removing a task word lowers the
  question's intent score. No word lists are involved.
- **Each translation's words for a subject** are looked up in its own vocabulary, embedded once
  (`vocabulary.npz` next to the index): *alcohol* → KJV *drinking, drunkenness, wine*, RV1909
  *bebida, embriaguez, borracho*. Candidates are ranked by how close the passages they occur
  in are to the subject, which drops look-alikes (*pork* → *swine*, not *porch*).

The example texts' embeddings are cached per embedder (`semantics-<embedder>.npz`; the
bundled embedder's copy ships in `storage/prebuilt/`).

## The three pipelines

**Direct questions** ("Where was Jesus born?") run a hybrid search in each translation:
embeddings and BM25 keywords, fused by reciprocal rank, because keywords find names
(*Goliath*) that embeddings blur. The best passages are then aligned across translations.
Passages are cut independently per version (KJV Matthew 2:1-3, RV1909 Mateo 2:1-4), so the
chunk covering the same verses is looked up in each, and the sentence that answers is quoted
from each. An English question thus shows Mateo 2:1 next to Matthew 2:1, although the Spanish
keyword index could never match it.

**Contradiction questions** ("Are there any paradoxes in the Bible according to alcohol?")
find the topic and its words in each translation, gather the passages that name it plus a
semantic search for each of two opposite statements, decide each passage's stance and show
both sides in every translation (details below). The LLM is then asked whether the passages
really contradict each other or whether their context explains it: who speaks and to whom,
law, narrative or poetry, moderation or excess.

**Translation questions** compare the same passages side by side: the verses cited ("How do
the translations render Psalm 16:10?", "Juan 3,16"), the first passages of a chapter named
without verses ("Compare the translations of Psalm 23") or, for a topic ("Do the translations
differ on hell?"), the passages where the versions word it with different concepts (Isaiah
14:9: KJV "Hell", RV1909 "sepulcro", WEB "Sheol"). Passages whose versions use the most
distinct concepts come first. Without a reference or a known topic, the passages found by a
direct search on the question's subject are compared.

## Contradictions: the two sides

- **The two statements.** A contradiction question is searched as its positive and negative
  form. Lexicon topics carry curated pairs ("wine is good… makes the heart of man glad" /
  "wine is a mocker… be not drunk"). For other topics a capable LLM writes the pair, which is
  accepted only if the models see the two as opposite. Without one, the subject is moved to
  both ends of the opposition axis.
- **The opposition axis.** 21 pairs of approving and condemning sentences, mostly in the
  Bible's own register ("Blessed is the man that…" / "Woe unto him that…", "It is lawful and
  clean" / "It is an abomination unto the LORD"), in several languages, define a direction in
  the embedding space. A passage's position on it says how approving or condemning it is,
  whatever its words. The axis orders LLM-written pairs, builds the two poles of unknown
  topics, and contributes to each passage's stance, weighted by how much the topic's two
  statements themselves differ along it (wine good / wine a mocker: a lot; faith / works:
  little).
- **Stance of a passage.** The clause about the topic is compared with both statements and
  placed on the axis, averaged over translations. Each side is then ranked by closeness to its
  own statement and relevance to the topic, one passage per chapter.
- **NLI check.** A natural-language-inference model tells whether a sentence entails or
  contradicts another, which embeddings can't. When it is available it re-checks the best
  candidates of both sides against both statements (which settles their side), verifies that
  an LLM-written pair really is contradictory, and flags parallel translations that contradict
  each other ("Meaning may differ: KJV and WEB contradict each other here"). Without it the
  sides are estimated from the embeddings, and the answer says so.
- **The original words of each side.** The notes count the Hebrew or Greek words behind each
  side and say whether both use the same one ("Both sides use H3196 yayin: the tension is in
  the original text, not a translation choice").

## Translations and the original languages

- **Parallel passages.** For each passage found, every other loaded version contributes the
  chunk covering the same verses. Where several overlap, the one closest in meaning to the
  found clause is taken; for the originals, the one sharing most verses.
- **Key clauses.** In each version, the clause that matters is chosen: the one closest to the
  question (or to a statement) in the first version, then its counterpart in the others, by
  meaning across languages and by wording within one language. Pairs of adjacent clauses are
  candidates too, since translations punctuate differently.
- **Verse for verse.** Passages keep where each verse starts, so the verses of the key clause
  are known and exactly those verses are shown from the Hebrew or Greek.
- **Wording notes.** The concept each version uses for the topic ("hell" / "sepulcro" / "Sheol"),
  and for two versions in the same language a word diff that keeps meaning changes ("charity" →
  "love", "did tempt" → "tested", "easily" dropped) apart from archaic-to-modern style ("thou"
  → "you").
- **The original word.** Each KJV word carries its Strong's number. With the Strong's
  dictionary, a note names it as written in the verse: "hell" = H7585 לִשְׁאוֹל *she'ôl* (KJV
  elsewhere: grave, pit), G1067 γέενναν *géenna* in Matthew 5:22, G86 ᾅδης *háidēs* in
  Revelation 20:14.
- **Searching the originals.** The bundled embedder doesn't know Hebrew or Greek, so the
  originals are shown next to what is found rather than searched by meaning; their embeddings
  are left empty, which costs nothing. `bible-rag search --versions wlc יין` searches them by
  keyword. An embedder that knows them (e.g. `sentence-transformers/LaBSE`) searches them too.
- **Aramaic.** The Biblical Aramaic of Daniel and Ezra is part of the WLC. For the New
  Testament, Etheridge's translation renders the Syriac Peshitta ("Meshicha", "shiul",
  "gihano"); the Syriac text itself isn't available as a free download.

## The lexicon

`bible_rag/lexicon.py` holds 16 topics: alcohol, money, divorce, war, hell, love and hate,
faith and works, seeing God, temptation, God repenting, oaths, anger, slavery, food (clean and
unclean), sabbath and lying. Each topic has:

- example phrases, matched by meaning when routing a question to it;
- its concepts and the words each language uses for them, to collect the passages that name
  it and to detect wording differences. Terms are matched on lowercase, accent-free text:
  `word*` is a prefix, `word?` a homograph that only counts when a parallel passage in another
  language confirms it (Spanish "vino" is also "came");
- the two opposite statements searched for and the titles of the two sides;
- verified notes about traps: RV1909 "alcohol" is *kohl*, eye paint (2 Reyes 9:30); KJV "meat"
  means any food; KJV "hell" renders Sheol, Hades and Gehenna alike. Passages a note cites
  become sources, so they can be checked.

To add a topic, add a `Topic` to `TOPICS` and check every term against the texts. Topics
outside the lexicon still work, by meaning alone (see [Quality](#quality)).

## Writing and checking the answer

The prompt depends on the kind of question. Each gives the LLM the passages grouped as they
will be shown, the translation notes, and the task: answer and cite; weigh whether the two
groups really contradict each other; or compare the renderings and say whether the
difference changes the sense. The answer is written in the question's language (English,
Spanish, Portuguese, French, German or Italian, identified by function words).

The model's words are then checked (`bible_rag/grounding.py`). Every reference ("John 3:16",
"Juan 3:16", "1 Cor. 13:4") must point into a retrieved passage, and every quote must appear in
one, ignoring case, accents and punctuation. Anything else is listed as unverified, and the
passages are shown in full.

A model that can't write a given answer is not used for it, and not even loaded: the bundled
SmolLM2 only writes answers to direct questions in English, with a short prompt and three
English passages. For the other answers the assistant's findings are shown as they are.

## Texts and passages

- **Exact verses.** SWORD modules carry the book, chapter and verse structure: the KJV has
  31,102 verses, matching the canon. Only the 66-book canon is indexed.
- **Faithful text.** Footnotes, cross-references, editorial headings and the WEB glossary are
  removed; canonical Psalm titles are kept. Where the source drops the space around italicised
  words, it is restored ("The LORD is"), while Spanish enclitics are joined ("díjole",
  "sácalo"). The KJV is the Strong's-numbered edition, which writes a few compounds apart
  ("can not", "where in").
- **Passages** are consecutive verses of one chapter, up to 450 characters with an 80-character
  overlap, which fits the 128-token window of MiniLM-style embedders. Each passage records
  where every verse starts.
- **RVR1960.** The PDF's page headers give the verse range and the margin numbers mark the line
  where each verse starts, so references are accurate to the line (`bible_rag/corpus.py`).

## Search indexes

One index per text, in `<embedder>/<text>/` under `storage/prebuilt/` (shipped) or
`storage/index/` (built locally, preferred when current). A text has its own index because a
multilingual embedder still prefers same-language passages: searching each text separately
gives every answer both languages.

| File | Content |
|---|---|
| `manifest.json` | embedder, source, license, fingerprint, format version |
| `passages.jsonl.gz` | the passages with their verse offsets |
| `vectors.npz` | embeddings as int8 with one scale per vector, and a hash of each passage's text |
| `bm25.npz` | the keyword index as arrays (loaded as is, not rebuilt) |
| `tags.json.gz` | Strong's numbers per word (KJV, WLC, SBLGNT) |
| `vocabulary.npz` | the text's words, embedded (first text of each language) |

int8 vectors take a quarter of the space of float32. On the evaluation queries they give the
same top result for every query, 99 % of the top 10 is identical, and the cosine error stays
below 0.002. Search is exact: a matrix product over at most 12,000 vectors takes about 2 ms.

**Fingerprints.** Each manifest records a hash of the source file (its pinned SHA-256), the
code that turns it into passages (`corpus.py`, `sources.py` and the keyword tokenizer, hashed
from the syntax tree so comments and docstrings don't count), the passage size and overlap, and
the selected books. At start-up the fingerprints are recomputed without reading the texts. An
index that doesn't match is still used, and answers carry a note until `bible-rag ingest`
rebuilds it.

**Incremental rebuilds.** A rebuild reuses the stored embedding of every passage whose text
hash is unchanged, and the vocabulary's embeddings word by word. Rebuilding all eight texts
therefore takes about as long as reading them (1-2 minutes) instead of embedding 78,000
passages (10-15 minutes on a CPU). Files are written with fixed timestamps and the stored int8
codes are reused as they are, so rebuilding unchanged data produces identical files. A lock
file keeps two processes from building at the same time.

**At start-up** a fresh copy loads the bundled embedder (about 3 s, once per process) and the
indexes; a question then takes 2-6 s without a language model.

## Quality

`bible-rag eval` asks 16 questions about well-known passages, each in English and Spanish, and
scores the results against the chapters that answer them. Measured on KJV + RV1909 with the
bundled embedder, k = 3:

| | English | Spanish |
|---|---|---|
| correct chapter retrieved, hybrid search (default) | 75 % | 75 % |
| same, embeddings only (`BIBLE_RAG_RETRIEVAL=dense`) | 62.5 % | 56.3 % |
| answer cites a correct passage, `extractive` (no LLM) | 10 / 16, nothing unverified | 11 / 16, nothing unverified |
| answer cites a correct passage, `smollm2` | 6 / 16, 4 with unverified claims | (English only) |

The bundled 135M model is worse than quoting the passages: with "Make thee an ark of gopher
wood" in front of it, it answered "cedar". Aligning the parallel passages lifts how often the
shown passages include the right chapter in the other language: English question → RV1909
passage from 31 % to 69 %, Spanish question → KJV passage from 44 % to 69 %.

`bible-rag eval --agent` measures the assistant itself, with the bundled embedder and no LLM
or NLI model. None of its questions is among the routing or topic examples:

| | |
|---|---|
| routing of 37 questions in English, Spanish, Portuguese, French, German and Italian | 100 % |
| topic of 40 questions (26 lexicon topics in paraphrase, 14 subjects outside the lexicon) | 100 % |
| contradiction questions on six topics (alcohol, war, food, faith and works, divorce, seeing God): share of the passages on each side that are well-known passages for that side, k = 3 | 78 % |
| the same six, ignoring the lexicon (meaning only, generic statements) | 19 % |

The last line is the hard case. Without the lexicon's statements, an LLM to write them or an
NLI model, a generic "X is good" / "X is evil" can't express "faith vs works" or "seeing God
vs no one can see God". The LLM and the NLI model could not be measured in the offline
environment where these figures were taken: run `bible-rag eval --agent` where they are
available. The planner's weights and thresholds were tuned on the bundled embedder; re-check
with the same command after changing models.
