import React, { useState, useEffect } from 'react';
import { Header, ActiveTab } from './components/Header';
import { TheologicalQAView } from './components/TheologicalQAView';
import { SemanticSearchView } from './components/SemanticSearchView';
import { VerseComparisonView } from './components/VerseComparisonView';
import { ArchitectureGuideView } from './components/ArchitectureGuideView';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('qa');
  const [selectedVerseIdForCompare, setSelectedVerseIdForCompare] = useState<string | null>(null);

  // Dark mode state with system preference fallback and localStorage persistence
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('scriptura_theme');
      if (saved) return saved === 'dark';
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add('dark');
      localStorage.setItem('scriptura_theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('scriptura_theme', 'light');
    }
  }, [isDark]);

  const handleToggleTheme = () => {
    setIsDark((prev) => !prev);
  };

  const handleSelectVerseForCompare = (verseId: string) => {
    setSelectedVerseIdForCompare(verseId);
    setActiveTab('compare');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleAskQuestionAboutPassage = (question: string) => {
    setActiveTab('qa');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    // Trigger the search query in TheologicalQAView
    setTimeout(() => {
      const input = document.querySelector('input[type="text"]') as HTMLInputElement;
      if (input) {
        input.value = question;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        const form = input.closest('form');
        if (form) form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    }, 100);
  };

  return (
    <div className="min-h-screen flex flex-col bg-stone-50 dark:bg-stone-950 text-stone-900 dark:text-stone-100 transition-colors">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isDark={isDark}
        onToggleTheme={handleToggleTheme}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {activeTab === 'qa' && (
          <TheologicalQAView onSelectVerseForCompare={handleSelectVerseForCompare} />
        )}

        {activeTab === 'search' && (
          <SemanticSearchView
            onSelectVerseForCompare={handleSelectVerseForCompare}
            onAskQuestionAboutPassage={handleAskQuestionAboutPassage}
          />
        )}

        {activeTab === 'compare' && (
          <VerseComparisonView
            initialVerseId={selectedVerseIdForCompare}
            onAskQuestionAboutPassage={handleAskQuestionAboutPassage}
          />
        )}

        {activeTab === 'architecture' && <ArchitectureGuideView />}
      </main>

      {/* Clean, Dignified Footer */}
      <footer className="border-t border-stone-200 dark:border-stone-800 bg-white/70 dark:bg-stone-900/70 py-8 px-4 sm:px-6 lg:px-8 text-xs text-stone-500 dark:text-stone-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-serif font-bold text-stone-800 dark:text-stone-200">
              Scriptura RAG
            </span>
            <span>·</span>
            <span>King James Version (KJV) & Reina-Valera 1960 (RVR1960)</span>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <button
              onClick={() => setActiveTab('architecture')}
              className="hover:text-stone-900 dark:hover:text-stone-200 transition-colors"
            >
              RAG Knowledge Base Blueprint
            </button>
            <span>·</span>
            <button
              onClick={() => setActiveTab('compare')}
              className="hover:text-stone-900 dark:hover:text-stone-200 transition-colors"
            >
              Parallel Reader
            </button>
            <span>·</span>
            <span>Public Domain & Canonical Textus Receptus</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
