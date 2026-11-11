import { BibleVerse, ChunkingStrategy, CorpusStats, ScriptureChunk, SearchResult } from '../types/bible';

/**
 * Computes cosine similarity between two numeric vectors.
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  const len = Math.min(vecA.length, vecB.length);
  for (let i = 0; i < len; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;
  return Math.max(0, Math.min(1, dotProduct / denominator));
}

/**
 * Normalizes text and splits into lowercase alphanumeric tokens.
 * Handles English and Spanish diacritics smoothly.
 */
const STOP_WORDS = new Set([
  'the', 'and', 'for', 'that', 'with', 'from', 'this', 'unto', 'they', 'them', 'shall', 'his', 'her', 'which',
  'los', 'las', 'del', 'por', 'con', 'que', 'para', 'una', 'uno', 'sus', 'como', 'mas', 'pero', 'ella', 'ellos'
]);

export function tokenize(text: string, filterStopWords: boolean = true): string[] {
  if (!text) return [];
  const tokens = text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 2);

  return filterStopWords ? tokens.filter((t) => !STOP_WORDS.has(t)) : tokens;
}

/**
 * Semantic concept keyword mappings for cross-lingual query vector extraction.
 * 16 dimensions corresponding to our canonical biblical semantic clusters:
 * 0: Creation / Origins
 * 1: Sovereignty / Power of God
 * 2: Fall / Sin / Suffering
 * 3: Law / Righteousness / Justice
 * 4: Redemption / Atonement / Cross
 * 5: Faith / Trust / Belief
 * 6: Love / Charity / Grace
 * 7: Wisdom / Truth / Guidance
 * 8: Holy Spirit / Presence / Comfort
 * 9: Prayer / Worship / Praise
 * 10: Peace / Anxiety / Refuge
 * 11: Hope / Eternal Life / Eschatology
 * 12: Spiritual Warfare / Armor / Victory
 * 13: Discipleship / Holiness / Obedience
 * 14: Christology / The Word / Incarnation
 * 15: Covenant / Kingdom / Blessing
 */
const SEMANTIC_CLUSTERS: { keywords: string[]; dim: number }[] = [
  { dim: 0, keywords: ['create', 'creation', 'beginning', 'earth', 'heaven', 'heavens', 'waters', 'creo', 'creacion', 'principio', 'tierra', 'cielos', 'origen', 'hacedor', 'luz', 'light'] },
  { dim: 1, keywords: ['god', 'lord', 'almighty', 'sovereign', 'power', 'father', 'omnipotente', 'altisimo', 'poder', 'soberano', 'fuerza', 'senor', 'jehova', 'dios', 'reina', 'reino'] },
  { dim: 2, keywords: ['sin', 'fall', 'death', 'iniquity', 'transgression', 'mourn', 'sorrow', 'crying', 'pain', 'pecado', 'muerte', 'iniquidad', 'rebelion', 'llanto', 'dolor', 'descarriamos', 'tristeza'] },
  { dim: 3, keywords: ['righteousness', 'justice', 'law', 'breastplate', 'judgment', 'justicia', 'ley', 'coraza', 'recto', 'mandamiento', 'justo'] },
  { dim: 4, keywords: ['atonement', 'cross', 'wound', 'bruised', 'stripes', 'healed', 'blood', 'sacrifice', 'salvation', 'save', 'saved', 'expiacion', 'cruz', 'herido', 'llaga', 'curados', 'sangre', 'sacrificio', 'salvo', 'salvacion'] },
  { dim: 5, keywords: ['faith', 'believe', 'trust', 'confidence', 'substance', 'evidence', 'lean', 'fe', 'creer', 'confiar', 'certeza', 'conviccion', 'fiate', 'apoyes', 'cree'] },
  { dim: 6, keywords: ['love', 'charity', 'kind', 'grace', 'mercy', 'goodness', 'patience', 'amor', 'caridad', 'benigno', 'gracia', 'misericordia', 'bondad', 'sufrido'] },
  { dim: 7, keywords: ['wisdom', 'truth', 'understanding', 'paths', 'direct', 'mind', 'verbo', 'sabiduria', 'verdad', 'prudencia', 'veredas', 'entendimiento', 'conocimiento', 'guiara'] },
  { dim: 8, keywords: ['spirit', 'holy', 'comfort', 'presence', 'dwell', 'soul', 'espiritu', 'consolador', 'presencia', 'habita', 'alma', 'aliento'] },
  { dim: 9, keywords: ['pray', 'prayer', 'supplication', 'thanksgiving', 'hallowed', 'name', 'worship', 'orar', 'oracion', 'ruego', 'accion de gracias', 'santificado', 'alabanza', 'peticiones'] },
  { dim: 10, keywords: ['peace', 'anxiety', 'afraid', 'troubled', 'refuge', 'fortress', 'rest', 'still', 'valley', 'shadow', 'paz', 'afan', 'ansiedad', 'miedo', 'temor', 'refugio', 'castillo', 'reposo', 'sombra'] },
  { dim: 11, keywords: ['hope', 'eternal', 'life', 'heaven', 'everlasting', 'resurrection', 'dwell', 'tear', 'lagrima', 'esperanza', 'vida eterna', 'cielo', 'morare', 'nuevas'] },
  { dim: 12, keywords: ['armor', 'armour', 'warfare', 'shield', 'darts', 'devil', 'wiles', 'conqueror', 'stand', 'armadura', 'guerra', 'escudo', 'dardos', 'diablo', 'asechanzas', 'vencedores', 'firmes', 'enemigos'] },
  { dim: 13, keywords: ['walk', 'discipleship', 'holy', 'renewing', 'transformed', 'sacrifice', 'follow', 'salt', 'light', 'santidad', 'renovacion', 'transformaos', 'seguir', 'luz', 'sal', 'discipulo'] },
  { dim: 14, keywords: ['christ', 'jesus', 'son', 'word', 'logos', 'flesh', 'begotten', 'way', 'cristo', 'jesus', 'hijo', 'verbo', 'carne', 'unigenito', 'camino'] },
  { dim: 15, keywords: ['covenant', 'kingdom', 'reward', 'blessed', 'blessing', 'shepherd', 'pastor', 'bienaventurados', 'reino', 'galardon', 'pacto', 'promesa'] },
];

/**
 * Extracts a normalized 16-D semantic vector from a natural language query in either English or Spanish.
 */
export function extractQueryVector(query: string): number[] {
  const tokens = tokenize(query);
  const vector = new Array(16).fill(0.05); // slight prior baseline

  if (tokens.length === 0) {
    return vector;
  }

  for (const cluster of SEMANTIC_CLUSTERS) {
    let matches = 0;
    for (const token of tokens) {
      if (cluster.keywords.some((k) => k === token || (k.length > 4 && (k.includes(token) || token.includes(k))))) {
        matches++;
      }
    }
    if (matches > 0) {
      vector[cluster.dim] += matches * 0.45;
    }
  }

  // Normalize
  const mag = Math.sqrt(vector.reduce((acc, v) => acc + v * v, 0));
  return mag === 0 ? vector : vector.map((v) => Number((v / mag).toFixed(4)));
}

/**
 * BM25 scoring for keyword / lexical relevance with document frequency map.
 */
export function bm25Score(
  queryTokens: string[],
  text: string,
  avgDocLen: number,
  totalDocs: number,
  docFrequencyMap?: Record<string, number>
): number {
  if (queryTokens.length === 0 || !text) return 0;
  const docTokens = tokenize(text);
  const docLen = docTokens.length;
  if (docLen === 0) return 0;

  const k1 = 1.2;
  const b = 0.75;
  let score = 0;

  const tokenCounts: Record<string, number> = {};
  for (const t of docTokens) {
    tokenCounts[t] = (tokenCounts[t] || 0) + 1;
  }

  for (const q of queryTokens) {
    const f = tokenCounts[q] || 0;
    if (f > 0) {
      const docFreq = docFrequencyMap?.[q] || 1;
      const idf = Math.max(0.1, Math.log((totalDocs - docFreq + 0.5) / (docFreq + 0.5) + 1));
      const numerator = f * (k1 + 1);
      const denominator = f + k1 * (1 - b + b * (docLen / (avgDocLen || 15)));
      score += idf * (numerator / denominator);
    }
  }

  return score;
}

/**
 * Performs hybrid semantic search with Reciprocal Rank Fusion (RRF).
 */
export function hybridSearch(
  query: string,
  corpus: BibleVerse[],
  options: {
    mode?: 'hybrid' | 'vector' | 'bm25';
    testament?: 'ALL' | 'OT' | 'NT';
    minScore?: number;
    topK?: number;
  } = {}
): SearchResult[] {
  const { mode = 'hybrid', testament = 'ALL', minScore = 0.1, topK = 15 } = options;
  const queryTokens = tokenize(query);
  const queryVec = extractQueryVector(query);

  // Filter corpus by testament if requested
  const filtered = testament === 'ALL' ? corpus : corpus.filter((v) => v.testament === testament);
  if (filtered.length === 0) return [];

  // Precompute document frequencies for query tokens
  const docFrequencyMap: Record<string, number> = {};
  for (const q of queryTokens) {
    let count = 0;
    for (const v of filtered) {
      const tokens = tokenize(`${v.kjv} ${v.rvr1960} ${v.bookNameEn} ${v.bookNameEs} ${v.theologicalThemes.join(' ')}`);
      if (tokens.includes(q)) count++;
    }
    docFrequencyMap[q] = count;
  }

  // Calculate average document length for BM25
  const totalWords = filtered.reduce((acc, v) => acc + tokenize(v.kjv + ' ' + v.rvr1960).length, 0);
  const avgDocLen = totalWords / filtered.length;

  // 1. Vector scores
  const vectorScores = filtered.map((verse) => {
    const sim = cosineSimilarity(queryVec, verse.semanticVector);
    return { verse, score: sim };
  });

  // 2. BM25 scores
  const bm25Scores = filtered.map((verse) => {
    const combinedText = `${verse.kjv} ${verse.rvr1960} ${verse.bookNameEn} ${verse.bookNameEs} ${verse.theologicalThemes.join(' ')}`;
    const score = bm25Score(queryTokens, combinedText, avgDocLen, filtered.length, docFrequencyMap);
    return { verse, score };
  });

  // If vector only:
  if (mode === 'vector') {
    return vectorScores
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((item) => ({
        verse: item.verse,
        score: Number(item.score.toFixed(4)),
        vectorScore: Number(item.score.toFixed(4)),
        bm25Score: 0,
        matchedTerms: findMatchedTerms(queryTokens, item.verse),
        crossLingualDistance: Number((1 - item.score).toFixed(4)),
      }));
  }

  // If BM25 only:
  if (mode === 'bm25') {
    return bm25Scores
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((item) => ({
        verse: item.verse,
        score: Number(item.score.toFixed(4)),
        vectorScore: 0,
        bm25Score: Number(item.score.toFixed(4)),
        matchedTerms: findMatchedTerms(queryTokens, item.verse),
        crossLingualDistance: 0.5,
      }));
  }

  // 3. Reciprocal Rank Fusion (RRF) for Hybrid Search:
  // RRF(d) = 1/(k + rank_vector) + 1/(k + rank_bm25), with standard k = 60
  const kRrf = 60;
  const sortedVector = [...vectorScores].sort((a, b) => b.score - a.score);
  const sortedBm25 = [...bm25Scores].sort((a, b) => b.score - a.score);

  const vectorRankMap = new Map<string, number>();
  sortedVector.forEach((item, idx) => vectorRankMap.set(item.verse.id, idx + 1));

  const bm25RankMap = new Map<string, number>();
  sortedBm25.forEach((item, idx) => bm25RankMap.set(item.verse.id, idx + 1));

  const results: SearchResult[] = filtered.map((verse) => {
    const vRank = vectorRankMap.get(verse.id) || filtered.length;
    const bRank = bm25RankMap.get(verse.id) || filtered.length;
    const vScore = vectorScores.find((x) => x.verse.id === verse.id)?.score || 0;
    const bScore = bm25Scores.find((x) => x.verse.id === verse.id)?.score || 0;

    // RRF score normalized to ~0-1 range
    const rrfScore = 1 / (kRrf + vRank) + (bScore > 0 ? 1 / (kRrf + bRank) : 0);
    // Weighted blended score combining direct cosine similarity and keyword boost
    const finalScore = Number((vScore * 0.7 + Math.min(1, bScore / 5) * 0.3).toFixed(4));

    return {
      verse,
      score: finalScore,
      vectorScore: Number(vScore.toFixed(4)),
      bm25Score: Number(bScore.toFixed(4)),
      matchedTerms: findMatchedTerms(queryTokens, verse),
      crossLingualDistance: Number((1 - vScore).toFixed(4)),
    };
  });

  return results
    .filter((r) => r.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function findMatchedTerms(queryTokens: string[], verse: BibleVerse): string[] {
  const verseText = tokenize(`${verse.kjv} ${verse.rvr1960} ${verse.bookNameEn} ${verse.bookNameEs} ${verse.theologicalThemes.join(' ')}`);
  return Array.from(new Set(queryTokens.filter((q) => verseText.includes(q))));
}

/**
 * Splits Scripture into structured chunks based on different chunking strategies:
 * 1. 'verse': 1 chunk per verse (fine granularity, loses broader context)
 * 2. 'pericope': 1 chunk per theological section/pericope (optimal contextual boundaries)
 * 3. 'sliding_window': overlapping 3-verse windows (captures neighboring context)
 */
export function chunkPassage(verses: BibleVerse[], strategy: ChunkingStrategy): ScriptureChunk[] {
  if (verses.length === 0) return [];

  if (strategy === 'verse') {
    return verses.map((v) => ({
      chunkId: `chunk-v-${v.id}`,
      strategy: 'verse',
      title: `${v.bookNameEn} ${v.chapter}:${v.verse}`,
      passageRange: `${v.bookNameEn} ${v.chapter}:${v.verse}`,
      verseCount: 1,
      estimatedTokens: estimateTokens(v.kjv + ' ' + v.rvr1960),
      contentKjv: `${v.verse}. ${v.kjv}`,
      contentRvr1960: `${v.verse}. ${v.rvr1960}`,
      verseIds: [v.id],
      parentPericopeId: v.pericopeEn,
    }));
  }

  if (strategy === 'pericope') {
    const grouped = new Map<string, BibleVerse[]>();
    for (const v of verses) {
      const key = `${v.bookId}.${v.chapter}.${v.pericopeEn}`;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(v);
    }

    const chunks: ScriptureChunk[] = [];
    grouped.forEach((vList) => {
      const first = vList[0];
      const last = vList[vList.length - 1];
      const range =
        first.verse === last.verse
          ? `${first.bookNameEn} ${first.chapter}:${first.verse}`
          : `${first.bookNameEn} ${first.chapter}:${first.verse}-${last.verse}`;

      const kjvText = vList.map((v) => `${v.verse}. ${v.kjv}`).join(' ');
      const rvrText = vList.map((v) => `${v.verse}. ${v.rvr1960}`).join(' ');

      chunks.push({
        chunkId: `chunk-p-${first.bookId}-${first.chapter}-${first.verse}`,
        strategy: 'pericope',
        title: first.pericopeEn,
        passageRange: range,
        verseCount: vList.length,
        estimatedTokens: estimateTokens(kjvText + ' ' + rvrText),
        contentKjv: kjvText,
        contentRvr1960: rvrText,
        verseIds: vList.map((v) => v.id),
      });
    });
    return chunks;
  }

  // Sliding window of 3 verses with 1 verse step
  const windowSize = 3;
  const chunks: ScriptureChunk[] = [];
  for (let i = 0; i < verses.length; i += 2) {
    const slice = verses.slice(i, i + windowSize);
    if (slice.length === 0) break;
    const first = slice[0];
    const last = slice[slice.length - 1];
    const range =
      first.verse === last.verse
        ? `${first.bookNameEn} ${first.chapter}:${first.verse}`
        : `${first.bookNameEn} ${first.chapter}:${first.verse}-${last.verse}`;

    const kjvText = slice.map((v) => `${v.verse}. ${v.kjv}`).join(' ');
    const rvrText = slice.map((v) => `${v.verse}. ${v.rvr1960}`).join(' ');

    chunks.push({
      chunkId: `chunk-w-${first.id}-${last.id}`,
      strategy: 'sliding_window',
      title: `Window: ${range}`,
      passageRange: range,
      verseCount: slice.length,
      estimatedTokens: estimateTokens(kjvText + ' ' + rvrText),
      contentKjv: kjvText,
      contentRvr1960: rvrText,
      verseIds: slice.map((v) => v.id),
    });
  }

  return chunks;
}

/**
 * Quick token estimator (~1.3 tokens per whitespace word).
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  const wordCount = text.trim().split(/\s+/).length;
  return Math.ceil(wordCount * 1.35);
}

/**
 * Calculates token budget across chunks.
 */
export function calculateTokenBudget(chunks: ScriptureChunk[]): number {
  return chunks.reduce((acc, c) => acc + c.estimatedTokens, 0);
}

/**
 * Computes general statistics on the corpus.
 */
export function getCorpusStats(corpus: BibleVerse[]): CorpusStats {
  const books = new Set(corpus.map((v) => v.bookId));
  const pericopes = new Set(corpus.map((v) => v.pericopeEn));
  const ot = corpus.filter((v) => v.testament === 'OT').length;
  const nt = corpus.filter((v) => v.testament === 'NT').length;
  const totalTokens = corpus.reduce((acc, v) => acc + estimateTokens(v.kjv + ' ' + v.rvr1960), 0);

  return {
    totalVerses: corpus.length,
    totalBooks: books.size,
    oldTestamentCount: ot,
    newTestamentCount: nt,
    totalPericopes: pericopes.size,
    avgTokensPerVerse: Math.round(totalTokens / (corpus.length || 1)),
    vectorDimensions: corpus[0]?.semanticVector.length || 16,
  };
}
