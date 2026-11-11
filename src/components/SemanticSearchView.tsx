import React, { useState } from 'react';
import { Search, SlidersHorizontal, ArrowRight, Activity, BookOpen, Layers } from 'lucide-react';
import { BibleVerse, SearchResult } from '../types/bible';
import { BIBLE_CORPUS } from '../data/bibleCorpus';
import { hybridSearch, extractQueryVector } from '../utils/ragEngine';

interface SemanticSearchViewProps {
  onSelectVerseForCompare: (verseId: string) => void;
  onAskQuestionAboutPassage: (question: string) => void;
}

const SAMPLE_SEARCHES = [
  { label: 'The Lord is my shepherd in dark valleys', lang: 'EN' },
  { label: 'Jehová es mi pastor y nada me faltará', lang: 'ES' },
  { label: 'Peace of God guarding hearts in anxiety', lang: 'EN' },
  { label: 'Paz que sobrepasa todo entendimiento en la aflicción', lang: 'ES' },
  { label: 'He was wounded for our transgressions', lang: 'EN' },
  { label: 'Herido fue por nuestras rebeliones y pecados', lang: 'ES' },
  { label: 'Whole armor of God against the devil', lang: 'EN' },
  { label: 'Armadura de Dios y dardos de fuego', lang: 'ES' },
  { label: 'God so loved the world everlasting life', lang: 'EN' },
  { label: 'Es pues la fe la certeza de lo que se espera', lang: 'ES' },
];

const DIMENSION_NAMES = [
  'Origins', 'Sovereignty', 'Sin & Fall', 'Justice',
  'Atonement', 'Faith', 'Love & Grace', 'Wisdom',
  'Holy Spirit', 'Prayer', 'Peace & Refuge', 'Hope & Heaven',
  'Spiritual Warfare', 'Discipleship', 'Christology', 'Covenant'
];

export const SemanticSearchView: React.FC<SemanticSearchViewProps> = ({
  onSelectVerseForCompare,
  onAskQuestionAboutPassage,
}) => {
  const [query, setQuery] = useState('Peace of God guarding hearts in anxiety');
  const [mode, setMode] = useState<'hybrid' | 'vector' | 'bm25'>('hybrid');
  const [testament, setTestament] = useState<'ALL' | 'OT' | 'NT'>('ALL');
  const [minScore, setMinScore] = useState<number>(0.15);
  const [showVectorMap, setShowVectorMap] = useState<boolean>(true);

  // Perform search
  const results: SearchResult[] = hybridSearch(query, BIBLE_CORPUS, {
    mode,
    testament,
    minScore,
    topK: 15,
  });

  const queryVector = extractQueryVector(query);

  return (
    <div className="space-y-8">
      {/* Search Header */}
      <section className="max-w-4xl mx-auto space-y-4 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 text-xs font-medium text-amber-800 dark:text-amber-300">
          <Layers className="w-3.5 h-3.5" />
          Dense Vector + BM25 Lexical Hybrid Search
        </div>

        <h1 className="font-serif text-3xl sm:text-4xl font-medium tracking-tight text-stone-900 dark:text-stone-100">
          Cross-Lingual Semantic Scripture Retrieval
        </h1>

        <p className="text-sm sm:text-base text-stone-600 dark:text-stone-300 max-w-2xl mx-auto leading-relaxed">
          Type conceptual themes or exact phrases in English or Spanish. The vector space aligns theological concepts across both translations automatically.
        </p>

        {/* Search Bar */}
        <div className="relative max-w-2xl mx-auto mt-4">
          <div className="relative flex items-center shadow-sm rounded-2xl bg-white dark:bg-stone-900 border border-stone-300 dark:border-stone-700 focus-within:border-amber-600 dark:focus-within:border-amber-500 focus-within:ring-2 focus-within:ring-amber-500/20 transition-all">
            <Search className="w-5 h-5 ml-4 text-stone-400 shrink-0" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search concepts across KJV & RVR1960..."
              className="w-full px-4 py-3.5 bg-transparent text-sm sm:text-base text-stone-900 dark:text-stone-100 placeholder-stone-400 focus:outline-none"
            />
            {query && (
              <button
                onClick={() => setQuery('')}
                className="mr-3 px-2 py-1 text-xs text-stone-400 hover:text-stone-600 dark:hover:text-stone-200"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Quick Sample Queries */}
        <div className="flex flex-wrap items-center justify-center gap-1.5 pt-1">
          {SAMPLE_SEARCHES.map((sample, idx) => (
            <button
              key={idx}
              onClick={() => setQuery(sample.label)}
              className="px-2.5 py-1 text-xs text-stone-600 dark:text-stone-300 bg-white dark:bg-stone-900 hover:bg-stone-100 dark:hover:bg-stone-800 border border-stone-200 dark:border-stone-800 rounded-lg transition-colors flex items-center gap-1.5 shadow-2xs"
            >
              <span className="text-[10px] uppercase font-mono text-amber-700 dark:text-amber-400 font-semibold">
                {sample.lang}
              </span>
              <span>{sample.label}</span>
            </button>
          ))}
        </div>
      </section>

      {/* Control Strip & Vector Visualizer */}
      <section className="max-w-5xl mx-auto p-5 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Mode Selector */}
          <div className="flex items-center gap-1 p-1 bg-stone-100 dark:bg-stone-800/80 rounded-xl">
            <button
              onClick={() => setMode('hybrid')}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
                mode === 'hybrid'
                  ? 'bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100 shadow-xs font-semibold'
                  : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-200'
              }`}
            >
              Hybrid (Vector + BM25)
            </button>
            <button
              onClick={() => setMode('vector')}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
                mode === 'vector'
                  ? 'bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100 shadow-xs font-semibold'
                  : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-200'
              }`}
            >
              Vector Only
            </button>
            <button
              onClick={() => setMode('bm25')}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
                mode === 'bm25'
                  ? 'bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100 shadow-xs font-semibold'
                  : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-200'
              }`}
            >
              BM25 Keyword Only
            </button>
          </div>

          {/* Testament Filters */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-stone-500 dark:text-stone-400">Testament:</span>
            <div className="flex items-center gap-1 p-1 bg-stone-100 dark:bg-stone-800/80 rounded-xl">
              {(['ALL', 'OT', 'NT'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTestament(t)}
                  className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all ${
                    testament === t
                      ? 'bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100 shadow-xs font-semibold'
                      : 'text-stone-600 dark:text-stone-400 hover:text-stone-900'
                  }`}
                >
                  {t === 'ALL' ? 'Both' : t === 'OT' ? 'Old Test.' : 'New Test.'}
                </button>
              ))}
            </div>
          </div>

          {/* Toggle Vector Heatmap */}
          <button
            onClick={() => setShowVectorMap(!showVectorMap)}
            className="px-3 py-1.5 text-xs font-medium text-stone-600 dark:text-stone-300 hover:text-stone-900 rounded-xl border border-stone-200 dark:border-stone-700 hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors flex items-center gap-1.5"
          >
            <Activity className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
            <span>{showVectorMap ? 'Hide Vector Heatmap' : 'Show Vector Heatmap'}</span>
          </button>
        </div>

        {/* Similarity Threshold Slider */}
        <div className="flex items-center gap-3 pt-2 border-t border-stone-100 dark:border-stone-800 text-xs text-stone-500 dark:text-stone-400">
          <span className="shrink-0">Similarity Cutoff: {Math.round(minScore * 100)}%</span>
          <input
            type="range"
            min="0"
            max="0.8"
            step="0.05"
            value={minScore}
            onChange={(e) => setMinScore(parseFloat(e.target.value))}
            className="w-48 accent-amber-700 dark:accent-amber-500 cursor-pointer"
          />
          <span className="text-stone-400">({results.length} matches found)</span>
        </div>

        {/* 16-Dimensional Semantic Activation Spectrum */}
        {showVectorMap && (
          <div className="pt-3 border-t border-stone-100 dark:border-stone-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-stone-700 dark:text-stone-300">
                16-D Semantic Activation Spectrum (Latent Space Embedding)
              </span>
              <span className="text-stone-400">Unit Vector Normalized</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-1.5">
              {queryVector.map((val, idx) => {
                const pct = Math.min(100, Math.round(val * 100));
                const isHigh = pct > 25;
                return (
                  <div
                    key={idx}
                    className={`p-2 rounded-xl border text-center transition-all ${
                      isHigh
                        ? 'bg-amber-50 dark:bg-amber-950/60 border-amber-300 dark:border-amber-800'
                        : 'bg-stone-50/50 dark:bg-stone-800/40 border-stone-200/60 dark:border-stone-800'
                    }`}
                  >
                    <div className="text-[10px] text-stone-500 dark:text-stone-400 truncate font-medium">
                      {DIMENSION_NAMES[idx]}
                    </div>
                    <div className="font-mono text-xs font-bold text-stone-800 dark:text-stone-200 mt-1">
                      {val.toFixed(2)}
                    </div>
                    {/* Visual Bar */}
                    <div className="w-full bg-stone-200 dark:bg-stone-700 h-1 rounded-full mt-1.5 overflow-hidden">
                      <div
                        className={`h-full ${isHigh ? 'bg-amber-600 dark:bg-amber-400' : 'bg-stone-400'}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* Results List */}
      <section className="max-w-5xl mx-auto space-y-4">
        <div className="flex items-center justify-between text-xs text-stone-500 dark:text-stone-400 px-1">
          <span>Displaying Top Ranked Passages across Translations</span>
          <span>Ranked by Reciprocal Rank Fusion & Cosine Proximity</span>
        </div>

        {results.length === 0 ? (
          <div className="p-12 text-center rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 space-y-3">
            <p className="font-serif text-lg text-stone-700 dark:text-stone-300">
              No verses met the current similarity threshold ({Math.round(minScore * 100)}%).
            </p>
            <p className="text-xs text-stone-500 dark:text-stone-400">
              Try lowering the similarity cutoff or searching for related words like "peace", "amor", "shepherd", or "creación".
            </p>
            <button
              onClick={() => setMinScore(0.05)}
              className="px-4 py-2 text-xs font-medium text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/60 rounded-xl hover:bg-amber-100 transition-colors"
            >
              Lower threshold to 5%
            </button>
          </div>
        ) : (
          results.map((result) => (
            <article
              key={result.verse.id}
              className="p-6 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-2xs space-y-4 transition-all hover:border-amber-300 dark:hover:border-amber-800"
            >
              {/* Card Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-100 dark:border-stone-800 pb-3">
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="font-serif font-bold text-lg text-stone-900 dark:text-stone-100">
                      {result.verse.bookNameEn} {result.verse.chapter}:{result.verse.verse}
                    </span>
                    <span className="text-xs text-stone-400 dark:text-stone-500">/</span>
                    <span className="font-serif font-bold text-lg text-amber-800 dark:text-amber-400">
                      {result.verse.bookNameEs} {result.verse.chapter}:{result.verse.verse}
                    </span>
                  </div>
                  <div className="text-xs text-stone-500 dark:text-stone-400">
                    <span>{result.verse.pericopeEn}</span>
                    <span className="mx-1.5">·</span>
                    <span>{result.verse.testament === 'OT' ? 'Old Testament' : 'New Testament'}</span>
                  </div>
                </div>

                {/* Score Pills */}
                <div className="flex items-center gap-2">
                  <div className="px-3 py-1 rounded-xl bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 text-xs font-mono font-bold text-amber-900 dark:text-amber-300">
                    {Math.round(result.score * 100)}% Match
                  </div>
                  <div className="text-[11px] font-mono text-stone-400 dark:text-stone-500">
                    v: {result.vectorScore.toFixed(2)} · bm25: {result.bm25Score.toFixed(1)}
                  </div>
                </div>
              </div>

              {/* Side-by-Side Scripture Parallel */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* KJV */}
                <div className="p-4 rounded-xl bg-stone-50 dark:bg-stone-950/60 border border-stone-200/70 dark:border-stone-800/80 space-y-1">
                  <div className="flex items-center justify-between text-[11px] font-semibold text-stone-500 dark:text-stone-400 uppercase tracking-wider">
                    <span>King James Version (KJV)</span>
                    <span className="font-mono text-stone-400">EN</span>
                  </div>
                  <p className="font-serif text-base text-stone-900 dark:text-stone-100 leading-relaxed">
                    "{result.verse.kjv}"
                  </p>
                </div>

                {/* RVR1960 */}
                <div className="p-4 rounded-xl bg-stone-50 dark:bg-stone-950/60 border border-stone-200/70 dark:border-stone-800/80 space-y-1">
                  <div className="flex items-center justify-between text-[11px] font-semibold text-amber-800 dark:text-amber-400 uppercase tracking-wider">
                    <span>Reina-Valera 1960 (RVR1960)</span>
                    <span className="font-mono text-stone-400">ES</span>
                  </div>
                  <p className="font-serif text-base text-stone-900 dark:text-stone-100 leading-relaxed">
                    "{result.verse.rvr1960}"
                  </p>
                </div>
              </div>

              {/* Card Footer: Metadata & Actions */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-1 text-xs">
                <div className="flex items-center gap-1.5 text-stone-500 dark:text-stone-400">
                  <span className="font-medium">Themes:</span>
                  {result.verse.theologicalThemes.map((theme, i) => (
                    <React.Fragment key={i}>
                      {i > 0 && <span aria-hidden="true">·</span>}
                      <span>{theme}</span>
                    </React.Fragment>
                  ))}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() =>
                      onAskQuestionAboutPassage(
                        `What is the theological significance and application of ${result.verse.bookNameEn} ${result.verse.chapter}:${result.verse.verse}?`
                      )
                    }
                    className="px-3 py-1.5 rounded-xl border border-stone-200 dark:border-stone-700 text-stone-700 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors flex items-center gap-1"
                  >
                    <span>Ask AI About Verse</span>
                  </button>

                  <button
                    onClick={() => onSelectVerseForCompare(result.verse.id)}
                    className="px-3.5 py-1.5 rounded-xl bg-amber-800 hover:bg-amber-900 dark:bg-amber-700 dark:hover:bg-amber-600 text-white font-medium transition-colors flex items-center gap-1.5"
                  >
                    <span>Compare in Reader</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </article>
          ))
        )}
      </section>
    </div>
  );
};
