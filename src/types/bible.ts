export type Testament = 'OT' | 'NT';

export interface BibleVerse {
  id: string; // e.g. "JHN.3.16"
  bookId: string; // e.g. "JHN"
  bookNameEn: string;
  bookNameEs: string;
  testament: Testament;
  chapter: number;
  verse: number;
  kjv: string;
  rvr1960: string;
  pericopeEn: string;
  pericopeEs: string;
  theologicalThemes: string[];
  keywordsEn: string[];
  keywordsEs: string[];
  semanticVector: number[]; // Normalized dense representation
}

export interface SearchResult {
  verse: BibleVerse;
  score: number; // Combined hybrid score
  vectorScore: number;
  bm25Score: number;
  matchedTerms: string[];
  crossLingualDistance: number;
}

export type ChunkingStrategy = 'verse' | 'pericope' | 'sliding_window';

export interface ScriptureChunk {
  chunkId: string;
  strategy: ChunkingStrategy;
  title: string;
  passageRange: string;
  verseCount: number;
  estimatedTokens: number;
  contentKjv: string;
  contentRvr1960: string;
  verseIds: string[];
  parentPericopeId?: string;
}

export interface TheologicalQAQuery {
  question: string;
  targetTranslation?: 'both' | 'kjv' | 'rvr1960';
  preferredLanguage?: 'en' | 'es' | 'auto';
}

export interface TheologicalQAResponse {
  question: string;
  summary: string;
  detailedAnswer: string;
  theologicalInsights: string[];
  citations: {
    verseId: string;
    reference: string;
    kjv: string;
    rvr1960: string;
    contextExplanation: string;
  }[];
  keyPrinciples: string[];
  suggestedFollowUps: string[];
  isAiGenerated: boolean;
}

export interface CorpusStats {
  totalVerses: number;
  totalBooks: number;
  oldTestamentCount: number;
  newTestamentCount: number;
  totalPericopes: number;
  avgTokensPerVerse: number;
  vectorDimensions: number;
}
