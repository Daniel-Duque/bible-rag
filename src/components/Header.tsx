import React from 'react';
import { BookOpen, Moon, Sun, Search, HelpCircle, Columns, Cpu } from 'lucide-react';

export type ActiveTab = 'qa' | 'search' | 'compare' | 'architecture';

interface HeaderProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  isDark,
  onToggleTheme,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-white/90 dark:bg-stone-900/90 backdrop-blur-md border-b border-stone-200 dark:border-stone-800 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-8">
        {/* Zone 1: Brand wordmark */}
        <button
          onClick={() => onTabChange('qa')}
          className="flex items-center gap-3 text-left focus:outline-none group shrink-0"
        >
          <div className="w-9 h-9 rounded-xl bg-amber-100 dark:bg-amber-950/70 border border-amber-300 dark:border-amber-800/80 flex items-center justify-center text-amber-900 dark:text-amber-300 shadow-sm transition-transform group-hover:scale-105">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <span className="font-serif text-xl font-bold tracking-tight text-stone-900 dark:text-stone-100 block leading-none">
              Scriptura RAG
            </span>
            <span className="text-[11px] font-sans text-stone-500 dark:text-stone-400 block mt-0.5 tracking-wide">
              KJV · RVR1960 Dual Corpus
            </span>
          </div>
        </button>

        {/* Zone 2: Navigation Links */}
        <nav className="hidden md:flex items-center gap-1 sm:gap-2">
          <button
            onClick={() => onTabChange('qa')}
            className={`px-3.5 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'qa'
                ? 'bg-amber-50 dark:bg-amber-950/50 text-amber-900 dark:text-amber-200 shadow-xs border border-amber-200/80 dark:border-amber-800/60 font-semibold'
                : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-200 hover:bg-stone-100/70 dark:hover:bg-stone-800/60'
            }`}
          >
            <HelpCircle className="w-4 h-4 opacity-75" />
            Theological Q&A
          </button>

          <button
            onClick={() => onTabChange('search')}
            className={`px-3.5 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'search'
                ? 'bg-amber-50 dark:bg-amber-950/50 text-amber-900 dark:text-amber-200 shadow-xs border border-amber-200/80 dark:border-amber-800/60 font-semibold'
                : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-200 hover:bg-stone-100/70 dark:hover:bg-stone-800/60'
            }`}
          >
            <Search className="w-4 h-4 opacity-75" />
            Semantic Search
          </button>

          <button
            onClick={() => onTabChange('compare')}
            className={`px-3.5 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'compare'
                ? 'bg-amber-50 dark:bg-amber-950/50 text-amber-900 dark:text-amber-200 shadow-xs border border-amber-200/80 dark:border-amber-800/60 font-semibold'
                : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-200 hover:bg-stone-100/70 dark:hover:bg-stone-800/60'
            }`}
          >
            <Columns className="w-4 h-4 opacity-75" />
            Verse Comparison
          </button>

          <button
            onClick={() => onTabChange('architecture')}
            className={`px-3.5 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'architecture'
                ? 'bg-amber-50 dark:bg-amber-950/50 text-amber-900 dark:text-amber-200 shadow-xs border border-amber-200/80 dark:border-amber-800/60 font-semibold'
                : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-200 hover:bg-stone-100/70 dark:hover:bg-stone-800/60'
            }`}
          >
            <Cpu className="w-4 h-4 opacity-75" />
            Architecture & Schema
          </button>
        </nav>

        {/* Zone 3: Actions (Theme Toggle & Mobile Trigger) */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={onToggleTheme}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="p-2.5 rounded-xl border border-stone-200 dark:border-stone-800 text-stone-600 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-300" /> : <Moon className="w-4 h-4 text-stone-700" />}
          </button>

          <button
            onClick={() => onTabChange('qa')}
            className="hidden sm:inline-flex px-4 py-2 text-xs font-semibold uppercase tracking-wider text-white bg-amber-800 hover:bg-amber-900 dark:bg-amber-700 dark:hover:bg-amber-600 rounded-xl transition-colors shadow-xs"
          >
            Ask Scripture
          </button>
        </div>
      </div>

      {/* Mobile Navigation bar */}
      <div className="md:hidden flex items-center justify-around px-2 py-2 border-t border-stone-200 dark:border-stone-800 bg-stone-50/95 dark:bg-stone-900/95">
        <button
          onClick={() => onTabChange('qa')}
          className={`flex-1 py-1.5 text-xs text-center font-medium rounded-lg ${
            activeTab === 'qa'
              ? 'text-amber-900 dark:text-amber-300 font-semibold'
              : 'text-stone-600 dark:text-stone-400'
          }`}
        >
          Q&A
        </button>
        <button
          onClick={() => onTabChange('search')}
          className={`flex-1 py-1.5 text-xs text-center font-medium rounded-lg ${
            activeTab === 'search'
              ? 'text-amber-900 dark:text-amber-300 font-semibold'
              : 'text-stone-600 dark:text-stone-400'
          }`}
        >
          Search
        </button>
        <button
          onClick={() => onTabChange('compare')}
          className={`flex-1 py-1.5 text-xs text-center font-medium rounded-lg ${
            activeTab === 'compare'
              ? 'text-amber-900 dark:text-amber-300 font-semibold'
              : 'text-stone-600 dark:text-stone-400'
          }`}
        >
          Compare
        </button>
        <button
          onClick={() => onTabChange('architecture')}
          className={`flex-1 py-1.5 text-xs text-center font-medium rounded-lg ${
            activeTab === 'architecture'
              ? 'text-amber-900 dark:text-amber-300 font-semibold'
              : 'text-stone-600 dark:text-stone-400'
          }`}
        >
          RAG Lab
        </button>
      </div>
    </header>
  );
};
