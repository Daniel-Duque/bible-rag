import React, { useState } from 'react';
import { Cpu, Database, Split, Layers, Check, Copy, Network, ShieldCheck, Zap, BookOpen } from 'lucide-react';
import { ChunkingStrategy, ScriptureChunk } from '../types/bible';
import { BIBLE_CORPUS } from '../data/bibleCorpus';
import { chunkPassage, calculateTokenBudget } from '../utils/ragEngine';

export const ArchitectureGuideView: React.FC = () => {
  const [selectedStrategy, setSelectedStrategy] = useState<ChunkingStrategy>('pericope');
  const [copiedSchema, setCopiedSchema] = useState(false);
  const [copiedSql, setCopiedSql] = useState(false);
  const [copiedScript, setCopiedScript] = useState(false);

  // Sample passage for chunking simulator: Psalm 23
  const psalmVerses = BIBLE_CORPUS.filter((v) => v.bookId === 'PSA' && v.chapter === 23);
  const simulatedChunks: ScriptureChunk[] = chunkPassage(psalmVerses, selectedStrategy);
  const totalTokens = calculateTokenBudget(simulatedChunks);

  const JSON_SCHEMA_EXAMPLE = `{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "BibleRAGKnowledgeBaseItem",
  "type": "object",
  "properties": {
    "id": { "type": "string", "example": "JHN.3.16" },
    "canonical_citation": { "type": "string", "example": "John 3:16" },
    "book_id": { "type": "string", "example": "JHN" },
    "book_number": { "type": "integer", "example": 43 },
    "testament": { "type": "string", "enum": ["OT", "NT"] },
    "genre": { "type": "string", "enum": ["Gospel", "Epistle", "Poetry", "Law", "Prophecy", "Wisdom", "Apocalyptic"] },
    "chapter": { "type": "integer", "example": 3 },
    "verse": { "type": "integer", "example": 16 },
    "pericope": {
      "type": "object",
      "properties": {
        "id": { "type": "string", "example": "JHN.PERICOPE.3.1-21" },
        "title_en": { "type": "string", "example": "Nicodemus and the Love of God" },
        "title_es": { "type": "string", "example": "Jesús y Nicodemo" },
        "verse_start": { "type": "integer", "example": 1 },
        "verse_end": { "type": "integer", "example": 21 }
      }
    },
    "translations": {
      "type": "object",
      "properties": {
        "kjv": {
          "type": "string",
          "example": "For God so loved the world, that he gave his only begotten Son..."
        },
        "rvr1960": {
          "type": "string",
          "example": "Porque de tal manera amó Dios al mundo, que ha dado a su Hijo unigénito..."
        }
      },
      "required": ["kjv", "rvr1960"]
    },
    "theological_themes": {
      "type": "array",
      "items": { "type": "string" },
      "example": ["Salvation", "Grace", "Love of God", "Eternal Life"]
    },
    "cross_references": {
      "type": "array",
      "items": { "type": "string" },
      "example": ["ROM.5.8", "1JN.4.9", "EPH.2.4"]
    },
    "embeddings": {
      "type": "object",
      "properties": {
        "model": { "type": "string", "example": "gemini-embedding-2" },
        "dimensions": { "type": "integer", "example": 768 },
        "multilingual_vector": {
          "type": "array",
          "items": { "type": "number" },
          "description": "Normalized dense vector embedding representing combined bilingual semantic meaning."
        }
      }
    }
  },
  "required": ["id", "book_id", "chapter", "verse", "translations", "embeddings"]
}`;

  const SQL_SCHEMA_EXAMPLE = `-- PostgreSQL + pgvector Production Schema for Bible RAG
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE bible_verses (
    id VARCHAR(32) PRIMARY KEY,                  -- e.g. 'JHN.3.16'
    book_id VARCHAR(8) NOT NULL,                 -- e.g. 'JHN'
    chapter INT NOT NULL,
    verse INT NOT NULL,
    testament VARCHAR(2) NOT NULL,               -- 'OT' or 'NT'
    genre VARCHAR(32) NOT NULL,                  -- 'Gospel', 'Poetry', etc.
    pericope_id VARCHAR(64) NOT NULL,
    pericope_title_en VARCHAR(255) NOT NULL,
    pericope_title_es VARCHAR(255) NOT NULL,
    text_kjv TEXT NOT NULL,
    text_rvr1960 TEXT NOT NULL,
    theological_themes TEXT[] NOT NULL,
    tsv_search TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', text_kjv) ||
        to_tsvector('spanish', text_rvr1960) ||
        to_tsvector('simple', pericope_title_en || ' ' || pericope_title_es)
    ) STORED,
    embedding vector(768) NOT NULL               -- Gemini / multilingual embedding
);

-- HNSW Index for ultra-fast approximate nearest neighbor vector search
CREATE INDEX idx_bible_embedding_hnsw 
ON bible_verses USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN Index for BM25 / Full-Text Lexical Search
CREATE INDEX idx_bible_tsv 
ON bible_verses USING gin(tsv_search);

-- B-Tree index for pericope parent-child lookups
CREATE INDEX idx_bible_pericope 
ON bible_verses (book_id, chapter, pericope_id);`;

  const INGESTION_SCRIPT_EXAMPLE = `// TypeScript / Node.js Ingestion Pipeline Blueprint
import { GoogleGenAI } from '@google/genai';
import pg from 'pg';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const db = new pg.Pool({ connectionString: process.env.DATABASE_URL });

async function ingestScriptureCorpus(bilingualVerseRows: any[]) {
  console.log(\`Ingesting \${bilingualVerseRows.length} verses...\`);

  // Batch in chunks of 50 to respect embedding rate limits
  const BATCH_SIZE = 50;
  for (let i = 0; i < bilingualVerseRows.length; i += BATCH_SIZE) {
    const batch = bilingualVerseRows.slice(i, i + BATCH_SIZE);

    // 1. Prepare bilingual content for embedding
    // Combining KJV + RVR1960 ensures both languages anchor in the same latent space
    const textPrompts = batch.map(row => 
      \`[\${row.citation}] KJV: \${row.text_kjv} | RVR1960: \${row.text_rvr1960} | Themes: \${row.themes.join(', ')}\`
    );

    // 2. Compute Gemini dense embeddings
    const embedResult = await ai.models.embedContent({
      model: 'gemini-embedding-2',
      contents: textPrompts,
    });

    // 3. Insert into PostgreSQL with pgvector
    for (let j = 0; j < batch.length; j++) {
      const row = batch[j];
      const vector = embedResult.embeddings[j].values;

      await db.query(\`
        INSERT INTO bible_verses 
        (id, book_id, chapter, verse, testament, genre, pericope_id, pericope_title_en, pericope_title_es, text_kjv, text_rvr1960, theological_themes, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding
      \`, [
        row.id, row.bookId, row.chapter, row.verse, row.testament, row.genre,
        row.pericopeId, row.pericopeEn, row.pericopeEs,
        row.kjv, row.rvr1960, row.themes,
        \`[\${vector.join(',')}]\`
      ]);
    }

    console.log(\`Ingested \${i + batch.length} / \${bilingualVerseRows.length} verses\`);
  }
}`;

  return (
    <div className="max-w-5xl mx-auto space-y-12">
      {/* Intro Header */}
      <section className="text-center space-y-4 max-w-3xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 text-xs font-medium text-amber-800 dark:text-amber-300">
          <Database className="w-3.5 h-3.5" />
          Production Engineering Guide
        </div>

        <h1 className="font-serif text-3xl sm:text-4xl font-medium tracking-tight text-stone-900 dark:text-stone-100">
          How to Structure a Biblical RAG Knowledge Base
        </h1>

        <p className="text-sm sm:text-base text-stone-600 dark:text-stone-300 leading-relaxed font-sans">
          The Bible contains <strong className="text-stone-900 dark:text-stone-100 font-semibold">31,102 verses</strong> across 66 books, translated across thousands of languages.
          Building an effective RAG system requires mastering chunking granularity, multilingual embedding spaces, hybrid search, and parent-child retrieval.
        </p>
      </section>

      {/* Chapter 1: The Chunking Dilemma & Interactive Simulator */}
      <section className="p-6 sm:p-8 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Split className="w-5 h-5 text-amber-700 dark:text-amber-400" />
            <h2 className="font-serif text-2xl font-medium text-stone-900 dark:text-stone-100">
              01. The Scripture Chunking Dilemma
            </h2>
          </div>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            Why traditional fixed-token windowing destroys biblical semantics, and how to choose optimal boundaries.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl border border-stone-200 dark:border-stone-800 bg-stone-50/60 dark:bg-stone-950/40 space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-stone-700 dark:text-stone-300">
              1. Verse-Level Chunking
            </span>
            <p className="text-xs text-stone-600 dark:text-stone-400 leading-relaxed">
              <strong>Pros:</strong> Pinpoint precision for exact citations.
              <br />
              <strong>Cons:</strong> Loses critical context. Single verses like John 11:35 ("Jesus wept") or Psalm 23:2 lack the theological narrative when isolated.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-amber-300 dark:border-amber-800 bg-amber-50/50 dark:bg-amber-950/40 space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-amber-900 dark:text-amber-300">
              2. Pericope (Theological Unit)
            </span>
            <p className="text-xs text-stone-600 dark:text-stone-400 leading-relaxed">
              <strong>Pros (Recommended):</strong> Encapsulates complete parables, stories, or arguments (e.g. Beatitudes, Psalm 23). Optimal for LLM grounding.
              <br />
              <strong>Cons:</strong> Requires biblical metadata curation.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-stone-200 dark:border-stone-800 bg-stone-50/60 dark:bg-stone-950/40 space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-stone-700 dark:text-stone-300">
              3. Sliding Window (3-Verse)
            </span>
            <p className="text-xs text-stone-600 dark:text-stone-400 leading-relaxed">
              <strong>Pros:</strong> Smooth overlap preserves transition across sentences without manual boundary tagging.
              <br />
              <strong>Cons:</strong> Index bloat (3x storage multiplier) and partial duplication.
            </p>
          </div>
        </div>

        {/* Interactive Chunking Simulator */}
        <div className="pt-4 border-t border-stone-200 dark:border-stone-800 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="font-serif font-bold text-base text-stone-900 dark:text-stone-100">
                Interactive Chunking Simulator: Psalm 23
              </h3>
              <p className="text-xs text-stone-500">
                Switch strategies to see live chunk boundaries and estimated token budgets:
              </p>
            </div>

            {/* Strategy Switcher */}
            <div className="flex items-center gap-1 p-1 bg-stone-100 dark:bg-stone-800 rounded-xl">
              {(['verse', 'pericope', 'sliding_window'] as ChunkingStrategy[]).map((strat) => (
                <button
                  key={strat}
                  onClick={() => setSelectedStrategy(strat)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all capitalize ${
                    selectedStrategy === strat
                      ? 'bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100 shadow-xs font-bold'
                      : 'text-stone-600 dark:text-stone-400 hover:text-stone-900'
                  }`}
                >
                  {strat === 'sliding_window' ? 'Sliding Window' : strat}
                </button>
              ))}
            </div>
          </div>

          {/* Stats Ribbon */}
          <div className="grid grid-cols-3 gap-3 p-3 rounded-xl bg-stone-50 dark:bg-stone-950/60 border border-stone-200/80 dark:border-stone-800 text-center text-xs">
            <div>
              <span className="text-stone-500 dark:text-stone-400 block text-[11px]">Total Chunks</span>
              <span className="font-mono text-base font-bold text-stone-900 dark:text-stone-100">
                {simulatedChunks.length}
              </span>
            </div>
            <div>
              <span className="text-stone-500 dark:text-stone-400 block text-[11px]">Token Budget</span>
              <span className="font-mono text-base font-bold text-amber-700 dark:text-amber-400">
                ~{totalTokens} tokens
              </span>
            </div>
            <div>
              <span className="text-stone-500 dark:text-stone-400 block text-[11px]">Avg Chunk Length</span>
              <span className="font-mono text-base font-bold text-stone-900 dark:text-stone-100">
                {Math.round(totalTokens / (simulatedChunks.length || 1))} tokens
              </span>
            </div>
          </div>

          {/* Render Chunks Preview */}
          <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
            {simulatedChunks.map((chunk, idx) => (
              <div
                key={chunk.chunkId}
                className="p-3.5 rounded-xl bg-stone-50/70 dark:bg-stone-800/40 border border-stone-200/60 dark:border-stone-800 text-xs space-y-1.5"
              >
                <div className="flex items-center justify-between font-mono text-[11px] text-amber-800 dark:text-amber-400">
                  <span className="font-bold">
                    Chunk #{idx + 1}: {chunk.passageRange}
                  </span>
                  <span className="text-stone-400">
                    {chunk.verseCount} {chunk.verseCount === 1 ? 'verse' : 'verses'} · ~{chunk.estimatedTokens} tokens
                  </span>
                </div>
                <div className="font-serif text-stone-700 dark:text-stone-300 text-sm leading-relaxed">
                  <span className="font-bold text-stone-400 mr-1.5">EN:</span>
                  {chunk.contentKjv}
                </div>
                <div className="font-serif text-stone-700 dark:text-stone-300 text-sm leading-relaxed">
                  <span className="font-bold text-amber-700 mr-1.5">ES:</span>
                  {chunk.contentRvr1960}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Chapter 2: The Production JSON Schema */}
      <section className="p-6 sm:p-8 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-amber-700 dark:text-amber-400" />
              <h2 className="font-serif text-2xl font-medium text-stone-900 dark:text-stone-100">
                02. Canonical Knowledge Base JSON Schema
              </h2>
            </div>
            <p className="text-xs text-stone-500 dark:text-stone-400">
              The foundational document structure for indexing 31,102 verses across KJV and RVR1960.
            </p>
          </div>

          <button
            onClick={() => {
              navigator.clipboard.writeText(JSON_SCHEMA_EXAMPLE);
              setCopiedSchema(true);
              setTimeout(() => setCopiedSchema(false), 2000);
            }}
            className="px-3.5 py-1.5 rounded-xl border border-stone-200 dark:border-stone-700 text-xs font-medium text-stone-700 dark:text-stone-300 hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors flex items-center gap-1.5"
          >
            {copiedSchema ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedSchema ? 'Copied' : 'Copy Schema'}</span>
          </button>
        </div>

        <div className="relative rounded-xl overflow-hidden border border-stone-800 bg-stone-950 p-4 font-mono text-xs text-stone-200 max-h-96 overflow-y-auto leading-relaxed">
          <pre>{JSON_SCHEMA_EXAMPLE}</pre>
        </div>
      </section>

      {/* Chapter 3: Cross-Lingual Vector Alignment & Gemini Embeddings */}
      <section className="p-6 sm:p-8 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Network className="w-5 h-5 text-amber-700 dark:text-amber-400" />
            <h2 className="font-serif text-2xl font-medium text-stone-900 dark:text-stone-100">
              03. Cross-Lingual Vector Space Alignment
            </h2>
          </div>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            How modern multilingual embedding models (e.g. Gemini Embeddings) project English and Spanish into shared geometric vectors.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="space-y-3 text-xs sm:text-sm text-stone-600 dark:text-stone-300 leading-relaxed font-sans">
            <p>
              In traditional keyword search, querying in Spanish (<em className="text-stone-900 dark:text-stone-100">"El Señor es mi pastor"</em>) will <strong className="text-amber-800 dark:text-amber-400">never match</strong> the English text (<em className="text-stone-900 dark:text-stone-100">"The LORD is my shepherd"</em>) without translation pipelines.
            </p>
            <p>
              With dense multilingual embeddings (<code className="px-1.5 py-0.5 rounded bg-stone-100 dark:bg-stone-800 font-mono text-xs">gemini-embedding-2</code>), semantic concepts are encoded into a 768-dimensional latent space where cosine similarity between parallel verses approaches <strong className="text-emerald-700 dark:text-emerald-400 font-bold">&gt; 0.94</strong>.
            </p>
            <div className="p-3 rounded-xl bg-amber-50/70 dark:bg-amber-950/40 border border-amber-200/80 dark:border-amber-800/60 text-xs">
              <strong>Architecture Best Practice:</strong> Embed both translations in the document payload:
              <br />
              <code className="text-stone-800 dark:text-stone-200 font-mono text-[11px]">
                content = "[PSA.23.1] KJV: The LORD is my shepherd... | RVR1960: Jehová es mi pastor..."
              </code>
              <br />
              This bridges cross-lingual synonyms and cultural idioms seamlessly.
            </div>
          </div>

          {/* Visual Embedding Alignment Diagram */}
          <div className="p-6 rounded-2xl bg-stone-50 dark:bg-stone-950/70 border border-stone-200 dark:border-stone-800 space-y-4 text-center">
            <div className="text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
              Multilingual Shared Geometric Manifold
            </div>

            <div className="flex items-center justify-center gap-3">
              <div className="p-3 rounded-xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700 text-xs text-stone-800 dark:text-stone-200 shadow-2xs">
                <span className="font-bold text-amber-800 dark:text-amber-400 block">KJV English</span>
                "The Lord is my shepherd"
              </div>

              <div className="text-stone-400 font-mono text-xs font-bold">
                ⟷ <br /> <span className="text-emerald-600 dark:text-emerald-400">sim: 0.94</span>
              </div>

              <div className="p-3 rounded-xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700 text-xs text-stone-800 dark:text-stone-200 shadow-2xs">
                <span className="font-bold text-amber-800 dark:text-amber-400 block">RVR1960 Español</span>
                "Jehová es mi pastor"
              </div>
            </div>

            <div className="text-xs text-stone-500 dark:text-stone-400 pt-2 border-t border-stone-200 dark:border-stone-800">
              Query: <span className="font-semibold text-stone-900 dark:text-stone-100">"Quién me guía y cuida como oveja?"</span>
              <br />
              Cosine Distance to both &lt; 0.15 (Direct Dual Retrieval)
            </div>
          </div>
        </div>
      </section>

      {/* Chapter 4: Hybrid Search & Reciprocal Rank Fusion (RRF) */}
      <section className="p-6 sm:p-8 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-700 dark:text-amber-400" />
            <h2 className="font-serif text-2xl font-medium text-stone-900 dark:text-stone-100">
              04. Hybrid Search & Reciprocal Rank Fusion (RRF)
            </h2>
          </div>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            Why pure dense vector search fails on proper names, and how BM25 + Vector + RRF delivers 99%+ accuracy.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3 text-xs sm:text-sm text-stone-600 dark:text-stone-300 leading-relaxed font-sans">
            <p>
              Dense vector embeddings excel at <em>conceptual themes</em> (e.g., "forgiveness", "spiritual warfare", "fear of the future"), but they struggle with:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-xs">
              <li>Rare proper biblical names (e.g. <em>Melchizedek</em>, <em>Mephibosheth</em>, <em>Gethsemane</em>).</li>
              <li>Direct citation queries (e.g. <em>"John 3:16"</em> or <em>"Romanos 8"</em>).</li>
              <li>Exact phrase queries.</li>
            </ul>
            <p>
              By combining BM25 full-text lexical ranking with dense vector cosine similarity via <strong>Reciprocal Rank Fusion (RRF)</strong>, we obtain the best of both worlds:
            </p>
            <div className="p-3 rounded-xl bg-stone-100 dark:bg-stone-800 font-mono text-xs text-stone-800 dark:text-stone-200">
              RRF_Score(d) = 1 / (60 + Rank_Vector(d)) + 1 / (60 + Rank_BM25(d))
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-stone-50 dark:bg-stone-950/70 border border-stone-200 dark:border-stone-800 space-y-3">
            <h4 className="font-serif font-bold text-base text-stone-900 dark:text-stone-100">
              The 3-Stage Bible RAG Pipeline
            </h4>
            <div className="space-y-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700 flex items-center justify-between">
                <span>Stage 1: Multi-Query Retrieval</span>
                <span className="font-mono text-amber-700 dark:text-amber-400">BM25 + Vector</span>
              </div>
              <div className="p-2.5 rounded-lg bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700 flex items-center justify-between">
                <span>Stage 2: RRF Blending & Re-ranking</span>
                <span className="font-mono text-amber-700 dark:text-amber-400">Top 30 → Top 5</span>
              </div>
              <div className="p-2.5 rounded-lg bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700 flex items-center justify-between">
                <span>Stage 3: Grounded Theological LLM</span>
                <span className="font-mono text-amber-700 dark:text-amber-400">Gemini 3.8 Flash</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Chapter 5: PostgreSQL pgvector Database Schema & Ingestion Code */}
      <section className="p-6 sm:p-8 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-amber-700 dark:text-amber-400" />
              <h2 className="font-serif text-2xl font-medium text-stone-900 dark:text-stone-100">
                05. Production Database Schema (PostgreSQL + pgvector)
              </h2>
            </div>
            <p className="text-xs text-stone-500 dark:text-stone-400">
              Ready-to-deploy SQL migration script with HNSW vector indexing and full-text TSVector generation.
            </p>
          </div>

          <button
            onClick={() => {
              navigator.clipboard.writeText(SQL_SCHEMA_EXAMPLE);
              setCopiedSql(true);
              setTimeout(() => setCopiedSql(false), 2000);
            }}
            className="px-3.5 py-1.5 rounded-xl border border-stone-200 dark:border-stone-700 text-xs font-medium text-stone-700 dark:text-stone-300 hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors flex items-center gap-1.5"
          >
            {copiedSql ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedSql ? 'Copied' : 'Copy SQL Schema'}</span>
          </button>
        </div>

        <div className="rounded-xl overflow-hidden border border-stone-800 bg-stone-950 p-4 font-mono text-xs text-stone-200 max-h-80 overflow-y-auto leading-relaxed">
          <pre>{SQL_SCHEMA_EXAMPLE}</pre>
        </div>

        {/* Batch Ingestion Script */}
        <div className="pt-4 border-t border-stone-200 dark:border-stone-800 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-serif font-bold text-base text-stone-900 dark:text-stone-100">
              Batch Ingestion & Embedding Script (TypeScript)
            </h4>
            <button
              onClick={() => {
                navigator.clipboard.writeText(INGESTION_SCRIPT_EXAMPLE);
                setCopiedScript(true);
                setTimeout(() => setCopiedScript(false), 2000);
              }}
              className="px-3 py-1 text-xs font-medium border border-stone-200 dark:border-stone-700 rounded-lg hover:bg-stone-50 dark:hover:bg-stone-800 text-stone-700 dark:text-stone-300 flex items-center gap-1"
            >
              {copiedScript ? <Check className="w-3 h-3 text-emerald-600" /> : <Copy className="w-3 h-3" />}
              <span>{copiedScript ? 'Copied' : 'Copy Script'}</span>
            </button>
          </div>

          <div className="rounded-xl overflow-hidden border border-stone-800 bg-stone-950 p-4 font-mono text-xs text-stone-200 max-h-72 overflow-y-auto leading-relaxed">
            <pre>{INGESTION_SCRIPT_EXAMPLE}</pre>
          </div>
        </div>
      </section>
    </div>
  );
};
