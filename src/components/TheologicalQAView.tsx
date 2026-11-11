import React, { useState } from 'react';
import { Search, Sparkles, BookOpen, Copy, Check, ArrowRight, CornerDownRight, Share2 } from 'lucide-react';
import { TheologicalQAResponse } from '../types/bible';

interface TheologicalQAViewProps {
  onSelectVerseForCompare: (verseId: string) => void;
}

const POPULAR_QUESTIONS = [
  {
    en: 'What does it mean that the Lord is my shepherd in Psalm 23?',
    es: '¿Qué significa que Jehová es mi pastor en el Salmo 23?',
    topic: 'Providence & Comfort',
  },
  {
    en: 'How does the Bible define love in 1 Corinthians 13?',
    es: '¿Cómo define la Biblia el amor en 1 Corintios 13?',
    topic: 'Love & Agape',
  },
  {
    en: 'What is the promise of salvation by grace in John 3:16?',
    es: '¿Cuál es la promesa de salvación por gracia en Juan 3:16?',
    topic: 'Salvation & Grace',
  },
  {
    en: 'How can believers overcome anxiety according to Philippians 4 and Matthew 6?',
    es: '¿Cómo pueden los creyentes vencer la ansiedad según Filipenses 4 y Mateo 6?',
    topic: 'Peace & Anxiety',
  },
  {
    en: 'What is the whole armor of God in Ephesians 6?',
    es: '¿Qué es la armadura de Dios en Efesios 6?',
    topic: 'Spiritual Warfare',
  },
  {
    en: 'Why is Jesus called the Word (Logos) in John 1?',
    es: '¿Por qué se llama a Jesús el Verbo en Juan 1?',
    topic: 'Christology',
  },
];

export const TheologicalQAView: React.FC<TheologicalQAViewProps> = ({ onSelectVerseForCompare }) => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [qaResponse, setQaResponse] = useState<TheologicalQAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleAsk = async (questionText: string) => {
    if (!questionText.trim()) return;
    setIsLoading(true);
    setError(null);
    setQuery(questionText);

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText }),
      });

      if (!res.ok) {
        throw new Error(`Failed to process question (${res.status})`);
      }

      const data: TheologicalQAResponse = await res.json();
      setQaResponse(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'An error occurred while answering your question.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyCitation = (ref: string, kjv: string, rvr: string, id: string) => {
    const text = `${ref}\nKJV: "${kjv}"\nRVR1960: "${rvr}"`;
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center max-w-3xl mx-auto pt-4 sm:pt-8 space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 text-xs font-medium text-amber-800 dark:text-amber-300">
          <BookOpen className="w-3.5 h-3.5" />
          Theological Answers Grounded in Scripture
        </div>

        <h1 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-medium tracking-tight text-stone-900 dark:text-stone-100 leading-tight">
          Ask Scripture with Bilingual Precision
        </h1>

        <p className="text-base sm:text-lg text-stone-600 dark:text-stone-300 font-sans leading-relaxed">
          Pose your biblical, doctrinal, or life questions in English or Spanish. Our RAG engine searches parallel passages across the{' '}
          <strong className="font-semibold text-stone-900 dark:text-stone-100">King James Version</strong> and{' '}
          <strong className="font-semibold text-stone-900 dark:text-stone-100">Reina-Valera 1960</strong> to deliver contextually grounded citations.
        </p>

        {/* Search Input Bar */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAsk(query);
          }}
          className="relative max-w-2xl mx-auto mt-6"
        >
          <div className="relative flex items-center shadow-sm rounded-2xl bg-white dark:bg-stone-900 border border-stone-300 dark:border-stone-700 focus-within:border-amber-600 dark:focus-within:border-amber-500 focus-within:ring-2 focus-within:ring-amber-500/20 transition-all">
            <Search className="w-5 h-5 ml-4 text-stone-400 shrink-0" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., What does it mean that the Lord is my shepherd? / ¿Qué es la fe en Hebreos 11?"
              className="w-full px-4 py-3.5 bg-transparent text-sm sm:text-base text-stone-900 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="mr-2 px-5 py-2.5 bg-amber-800 hover:bg-amber-900 dark:bg-amber-700 dark:hover:bg-amber-600 disabled:opacity-50 text-white rounded-xl text-xs font-semibold uppercase tracking-wider transition-all shrink-0 flex items-center gap-1.5"
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5" />
              )}
              <span>Ask</span>
            </button>
          </div>
        </form>

        {/* Popular Questions Grid */}
        <div className="pt-4">
          <p className="text-xs font-medium uppercase tracking-widest text-stone-400 dark:text-stone-500 mb-3">
            Explore Common Theological Questions
          </p>
          <div className="flex flex-wrap items-center justify-center gap-2">
            {POPULAR_QUESTIONS.map((q, idx) => (
              <button
                key={idx}
                onClick={() => handleAsk(q.en)}
                className="px-3.5 py-1.5 text-xs text-stone-700 dark:text-stone-300 bg-white dark:bg-stone-900 hover:bg-amber-50 dark:hover:bg-amber-950/40 border border-stone-200 dark:border-stone-800 rounded-xl transition-colors text-left flex items-center gap-1.5 shadow-2xs"
              >
                <span className="text-amber-800 dark:text-amber-400 font-medium">·</span>
                <span>{q.en}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="max-w-4xl mx-auto p-8 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 space-y-6 animate-pulse">
          <div className="h-4 bg-stone-200 dark:bg-stone-800 rounded w-1/4" />
          <div className="h-8 bg-stone-200 dark:bg-stone-800 rounded w-3/4" />
          <div className="space-y-3">
            <div className="h-4 bg-stone-200 dark:bg-stone-800 rounded w-full" />
            <div className="h-4 bg-stone-200 dark:bg-stone-800 rounded w-5/6" />
            <div className="h-4 bg-stone-200 dark:bg-stone-800 rounded w-4/6" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
            <div className="h-32 bg-stone-100 dark:bg-stone-800/50 rounded-xl" />
            <div className="h-32 bg-stone-100 dark:bg-stone-800/50 rounded-xl" />
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="max-w-3xl mx-auto p-4 rounded-xl bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Active Response Display */}
      {qaResponse && !isLoading && (
        <article className="max-w-4xl mx-auto space-y-8 animate-fadeIn">
          {/* Header Card */}
          <div className="p-6 sm:p-8 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-100 dark:border-stone-800 pb-4 text-xs text-stone-500 dark:text-stone-400">
              <span className="font-medium text-amber-800 dark:text-amber-400 uppercase tracking-wider">
                Theological Response
              </span>
              <span>
                Dual Citations: KJV (English) · RVR1960 (Español)
              </span>
            </div>

            <h2 className="font-serif text-2xl sm:text-3xl font-medium text-stone-900 dark:text-stone-100">
              {qaResponse.question}
            </h2>

            {/* Executive Summary */}
            <div className="p-4 rounded-xl bg-amber-50/80 dark:bg-amber-950/40 border border-amber-200/80 dark:border-amber-800/60">
              <p className="text-sm sm:text-base font-serif italic text-stone-800 dark:text-stone-200 leading-relaxed">
                "{qaResponse.summary}"
              </p>
            </div>

            {/* Detailed Answer */}
            <div className="prose dark:prose-invert max-w-none text-stone-700 dark:text-stone-300 font-sans text-base leading-relaxed space-y-4 whitespace-pre-line pt-2">
              {qaResponse.detailedAnswer}
            </div>
          </div>

          {/* Dual-Translation Scripture Citations */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-serif text-xl sm:text-2xl font-medium text-stone-900 dark:text-stone-100">
                Biblical Grounding & Contextual Citations
              </h3>
              <span className="text-xs text-stone-500 dark:text-stone-400">
                {qaResponse.citations.length} Verified References
              </span>
            </div>

            <div className="grid grid-cols-1 gap-5">
              {qaResponse.citations.map((citation, idx) => (
                <div
                  key={idx}
                  className="rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 p-6 shadow-2xs space-y-4 transition-all hover:border-stone-300 dark:hover:border-stone-700"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-100 dark:border-stone-800 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="font-serif font-bold text-base text-stone-900 dark:text-stone-100">
                        {citation.reference}
                      </span>
                      <span className="text-xs text-stone-400 dark:text-stone-500">·</span>
                      <span className="text-xs text-stone-500 dark:text-stone-400">
                        Parallel Testimony
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() =>
                          handleCopyCitation(
                            citation.reference,
                            citation.kjv,
                            citation.rvr1960,
                            citation.verseId
                          )
                        }
                        className="px-2.5 py-1 text-xs font-medium text-stone-600 dark:text-stone-300 hover:text-stone-900 dark:hover:text-stone-100 rounded-lg hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors flex items-center gap-1"
                        title="Copy parallel verses"
                      >
                        {copiedId === citation.verseId ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-600" />
                            <span>Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>

                      <button
                        onClick={() => onSelectVerseForCompare(citation.verseId)}
                        className="px-2.5 py-1 text-xs font-medium text-amber-800 dark:text-amber-400 hover:text-amber-900 dark:hover:text-amber-300 rounded-lg hover:bg-amber-50 dark:hover:bg-amber-950/50 transition-colors flex items-center gap-1"
                        title="Compare full passage side-by-side"
                      >
                        <span>Compare in Reader</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Parallel Translation Columns */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* KJV Box */}
                    <div className="p-4 rounded-xl bg-stone-50 dark:bg-stone-950/60 border border-stone-200/70 dark:border-stone-800/80 space-y-1.5">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400 block">
                        King James Version (KJV)
                      </span>
                      <p className="font-serif text-stone-900 dark:text-stone-100 text-base leading-relaxed">
                        "{citation.kjv}"
                      </p>
                    </div>

                    {/* RVR1960 Box */}
                    <div className="p-4 rounded-xl bg-stone-50 dark:bg-stone-950/60 border border-stone-200/70 dark:border-stone-800/80 space-y-1.5">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-amber-800 dark:text-amber-400 block">
                        Reina-Valera 1960 (RVR1960)
                      </span>
                      <p className="font-serif text-stone-900 dark:text-stone-100 text-base leading-relaxed">
                        "{citation.rvr1960}"
                      </p>
                    </div>
                  </div>

                  {/* Context Explanation */}
                  <div className="text-xs text-stone-600 dark:text-stone-400 bg-stone-50/50 dark:bg-stone-800/30 p-3 rounded-lg border border-stone-100 dark:border-stone-800/50">
                    <strong className="font-semibold text-stone-700 dark:text-stone-300">Contextual Exegesis: </strong>
                    {citation.contextExplanation}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Theological Insights & Key Principles */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-6 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 space-y-3">
              <h4 className="font-serif text-lg font-medium text-stone-900 dark:text-stone-100">
                Theological Insights
              </h4>
              <ul className="space-y-2.5 text-sm text-stone-600 dark:text-stone-300">
                {qaResponse.theologicalInsights.map((insight, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-amber-800 dark:text-amber-400 font-bold mt-0.5">·</span>
                    <span>{insight}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="p-6 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 space-y-3">
              <h4 className="font-serif text-lg font-medium text-stone-900 dark:text-stone-100">
                Key Biblical Principles
              </h4>
              <ul className="space-y-2.5 text-sm text-stone-600 dark:text-stone-300">
                {qaResponse.keyPrinciples.map((principle, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-emerald-700 dark:text-emerald-400 font-bold mt-0.5">✓</span>
                    <span>{principle}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Suggested Follow-Ups */}
          {qaResponse.suggestedFollowUps && qaResponse.suggestedFollowUps.length > 0 && (
            <div className="p-6 rounded-2xl bg-amber-50/60 dark:bg-stone-900/60 border border-amber-200/60 dark:border-stone-800 space-y-3">
              <h4 className="text-xs font-semibold uppercase tracking-widest text-amber-900 dark:text-amber-400">
                Related Biblical Inquiries
              </h4>
              <div className="flex flex-col sm:flex-row flex-wrap gap-2 pt-1">
                {qaResponse.suggestedFollowUps.map((followUp, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAsk(followUp)}
                    className="text-xs sm:text-sm text-stone-800 dark:text-stone-200 hover:text-amber-900 dark:hover:text-amber-300 bg-white dark:bg-stone-800/80 px-3 py-2 rounded-xl border border-stone-200 dark:border-stone-700 text-left transition-colors flex items-center gap-2 group"
                  >
                    <CornerDownRight className="w-3.5 h-3.5 text-amber-700 dark:text-amber-400 transition-transform group-hover:translate-x-0.5" />
                    <span>{followUp}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </article>
      )}
    </div>
  );
};
