import React, { useState, useEffect } from 'react';
import { Columns, Copy, Check, Info, ArrowLeft, ArrowRight, BookOpen } from 'lucide-react';
import { BibleVerse } from '../types/bible';
import { BIBLE_CORPUS } from '../data/bibleCorpus';

interface VerseComparisonViewProps {
  initialVerseId?: string | null;
  onAskQuestionAboutPassage: (question: string) => void;
}

interface PassageGroup {
  id: string;
  bookId: string;
  chapter: number;
  labelEn: string;
  labelEs: string;
  pericopeEn: string;
  pericopeEs: string;
}

const PASSAGE_GROUPS: PassageGroup[] = [
  { id: 'GEN-1', bookId: 'GEN', chapter: 1, labelEn: 'Genesis 1 (Creation)', labelEs: 'Génesis 1 (La Creación)', pericopeEn: 'The Creation of Heaven and Earth', pericopeEs: 'La creación de los cielos y la tierra' },
  { id: 'GEN-3', bookId: 'GEN', chapter: 3, labelEn: 'Genesis 3 (The Fall & Promise)', labelEs: 'Génesis 3 (La Caída y la Promesa)', pericopeEn: 'The Protoevangelium', pericopeEs: 'El Protoevangelio' },
  { id: 'PSA-23', bookId: 'PSA', chapter: 23, labelEn: 'Psalm 23 (The Shepherd Psalm)', labelEs: 'Salmo 23 (Jehová es mi Pastor)', pericopeEn: 'The Shepherd of My Soul', pericopeEs: 'El Señor es mi pastor' },
  { id: 'PSA-91', bookId: 'PSA', chapter: 91, labelEn: 'Psalm 91 (Dwelling in the Secret Place)', labelEs: 'Salmo 91 (Bajo la sombra del Omnipotente)', pericopeEn: 'Security of the Faithful', pericopeEs: 'Seguridad del creyente' },
  { id: 'PRO-3', bookId: 'PRO', chapter: 3, labelEn: 'Proverbs 3 (Trust in the LORD)', labelEs: 'Proverbios 3 (Fíate de Jehová)', pericopeEn: 'Trust with All Thine Heart', pericopeEs: 'Confianza en el Señor' },
  { id: 'ISA-53', bookId: 'ISA', chapter: 53, labelEn: 'Isaiah 53 (The Suffering Servant)', labelEs: 'Isaías 53 (El Siervo Sufriente)', pericopeEn: 'Substitutionary Atonement', pericopeEs: 'Expiación sustitutiva' },
  { id: 'MAT-5', bookId: 'MAT', chapter: 5, labelEn: 'Matthew 5 (The Beatitudes)', labelEs: 'Mateo 5 (Las Bienaventuranzas)', pericopeEn: 'The Sermon on the Mount', pericopeEs: 'El Sermón del Monte' },
  { id: 'MAT-6', bookId: 'MAT', chapter: 6, labelEn: "Matthew 6 (Lord's Prayer & Do Not Worry)", labelEs: 'Mateo 6 (Padre Nuestro y el Afán)', pericopeEn: 'Kingdom Priorities', pericopeEs: 'Prioridades del Reino' },
  { id: 'JHN-1', bookId: 'JHN', chapter: 1, labelEn: 'John 1 (The Word Made Flesh)', labelEs: 'Juan 1 (El Verbo Encarnado)', pericopeEn: 'The Deity of the Word', pericopeEs: 'El Verbo era Dios' },
  { id: 'JHN-3', bookId: 'JHN', chapter: 3, labelEn: 'John 3 (God So Loved the World)', labelEs: 'Juan 3 (De tal manera amó Dios)', pericopeEn: 'The Golden Verse of Grace', pericopeEs: 'Salvación por fe' },
  { id: 'JHN-14', bookId: 'JHN', chapter: 14, labelEn: 'John 14 (The Way, Truth, and Life)', labelEs: 'Juan 14 (El Camino, la Verdad y la Vida)', pericopeEn: 'Peace and the Holy Spirit', pericopeEs: 'La paz de Cristo' },
  { id: 'ROM-8', bookId: 'ROM', chapter: 8, labelEn: 'Romans 8 (More Than Conquerors)', labelEs: 'Romanos 8 (Más que Vencedores)', pericopeEn: 'The Inseparable Love of God', pericopeEs: 'Inseparables del amor de Dios' },
  { id: 'ROM-12', bookId: 'ROM', chapter: 12, labelEn: 'Romans 12 (Living Sacrifice)', labelEs: 'Romanos 12 (Sacrificio Vivo)', pericopeEn: 'Christian Transformation', pericopeEs: 'Renovación del entendimiento' },
  { id: '1CO-13', bookId: '1CO', chapter: 13, labelEn: '1 Corinthians 13 (The Chapter of Love)', labelEs: '1 Corintios 13 (La Preeminencia del Amor)', pericopeEn: 'Charity Never Faileth', pericopeEs: 'El amor nunca deja de ser' },
  { id: 'EPH-6', bookId: 'EPH', chapter: 6, labelEn: 'Ephesians 6 (The Whole Armor of God)', labelEs: 'Efesios 6 (La Armadura de Dios)', pericopeEn: 'Spiritual Warfare', pericopeEs: 'La lucha espiritual' },
  { id: 'PHP-4', bookId: 'PHP', chapter: 4, labelEn: 'Philippians 4 (The Peace of God)', labelEs: 'Filipenses 4 (La Paz de Dios)', pericopeEn: 'Contentment and Strength', pericopeEs: 'Regocijo y paz en Cristo' },
  { id: 'HEB-11', bookId: 'HEB', chapter: 11, labelEn: 'Hebrews 11 (The Nature of Faith)', labelEs: 'Hebreos 11 (La Esencia de la Fe)', pericopeEn: 'The Evidence of Things Not Seen', pericopeEs: 'Certeza y convicción' },
  { id: 'REV-21', bookId: 'REV', chapter: 21, labelEn: 'Revelation 21 (New Heaven and New Earth)', labelEs: 'Apocalipsis 21 (Cielo Nuevo y Tierra Nueva)', pericopeEn: 'All Tears Wiped Away', pericopeEs: 'Enjugará Dios toda lágrima' },
];

const TRANSLATION_NOTES: Record<string, { termEn: string; termEs: string; note: string }> = {
  'PSA-23': {
    termEn: 'shall not want',
    termEs: 'nada me faltará',
    note: "KJV uses archaic 'want' in the sense of lack or deficiency, identical to RVR1960 'nada me faltará' (nothing will be lacking to me). Both reflect the Hebrew 'ehsar'.",
  },
  '1CO-13': {
    termEn: 'charity (agape)',
    termEs: 'el amor',
    note: "KJV renders Greek 'agape' as 'charity' (from Latin 'caritas', sacrificial divine love), distinct from mere emotional affection. RVR1960 renders it directly as 'el amor'.",
  },
  'JHN-1': {
    termEn: 'the Word (Logos)',
    termEs: 'el Verbo',
    note: "Both translations employ formal equivalence for Greek 'Logos'. RVR1960 translates it as 'el Verbo' (the active verbal articulation of God), while KJV translates it as 'the Word'.",
  },
  'ISA-53': {
    termEn: 'wounded / bruised',
    termEs: 'herido / molido',
    note: "Both translations convey physical and spiritual crushing: KJV 'wounded... bruised', RVR1960 'herido... molido'. Both capture the vicarious substitutionary nature of the Messiah's suffering.",
  },
  'PHP-4': {
    termEn: 'careful for nothing',
    termEs: 'por nada estéis afanosos',
    note: "KJV 'be careful for nothing' means 'do not be filled with anxious cares' (Greek: 'merimnate'). RVR1960 captures this with 'por nada estéis afanosos'.",
  },
  'HEB-11': {
    termEn: 'substance / evidence',
    termEs: 'certeza / convicción',
    note: "KJV 'substance' (underlying foundation, 'hypostasis') and 'evidence' ('elenchos') align closely with RVR1960 'certeza' (assurance) and 'convicción' (demonstrative proof of the unseen).",
  },
};

export const VerseComparisonView: React.FC<VerseComparisonViewProps> = ({
  initialVerseId,
  onAskQuestionAboutPassage,
}) => {
  const [selectedPassageId, setSelectedPassageId] = useState<string>('PSA-23');
  const [highlightVerseId, setHighlightVerseId] = useState<string | null>(null);
  const [copiedAll, setCopiedAll] = useState(false);

  // If initialVerseId is passed, locate matching passage
  useEffect(() => {
    if (initialVerseId) {
      const match = BIBLE_CORPUS.find((v) => v.id === initialVerseId);
      if (match) {
        const found = PASSAGE_GROUPS.find(
          (p) => p.bookId === match.bookId && p.chapter === match.chapter
        );
        if (found) {
          setSelectedPassageId(found.id);
          setHighlightVerseId(match.id);
        }
      }
    }
  }, [initialVerseId]);

  const currentGroup = PASSAGE_GROUPS.find((p) => p.id === selectedPassageId) || PASSAGE_GROUPS[0];

  const currentVerses = BIBLE_CORPUS.filter(
    (v) => v.bookId === currentGroup.bookId && v.chapter === currentGroup.chapter
  );

  const nuanceNote = TRANSLATION_NOTES[selectedPassageId];

  const handleCopyPassage = () => {
    const text = currentVerses
      .map(
        (v) =>
          `[${v.bookNameEn} ${v.chapter}:${v.verse}]\nKJV: ${v.kjv}\nRVR1960: ${v.rvr1960}`
      )
      .join('\n\n');
    navigator.clipboard.writeText(text);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2000);
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Passage Selector Ribbon */}
      <section className="p-6 rounded-2xl bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-800 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Columns className="w-5 h-5 text-amber-700 dark:text-amber-400" />
              <h2 className="font-serif text-2xl font-medium text-stone-900 dark:text-stone-100">
                Parallel Scripture Reader
              </h2>
            </div>
            <p className="text-xs text-stone-500 dark:text-stone-400">
              Synchronized verse-by-verse comparison between the Authorized King James Version (1611) and Reina-Valera 1960.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyPassage}
              className="px-3.5 py-2 rounded-xl text-xs font-medium border border-stone-200 dark:border-stone-700 hover:bg-stone-50 dark:hover:bg-stone-800 text-stone-700 dark:text-stone-300 transition-colors flex items-center gap-1.5"
            >
              {copiedAll ? (
                <>
                  <Check className="w-4 h-4 text-emerald-600" />
                  <span>Copied Entire Passage</span>
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  <span>Copy Parallel Passage</span>
                </>
              )}
            </button>

            <button
              onClick={() =>
                onAskQuestionAboutPassage(
                  `Provide a detailed comparative theological analysis of ${currentGroup.labelEn} between KJV and RVR1960.`
                )
              }
              className="px-4 py-2 rounded-xl text-xs font-semibold uppercase tracking-wider text-white bg-amber-800 hover:bg-amber-900 dark:bg-amber-700 dark:hover:bg-amber-600 transition-colors shadow-xs"
            >
              Exegesis with AI
            </button>
          </div>
        </div>

        {/* Passage Dropdown Selection */}
        <div className="pt-2 flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <label className="text-xs font-semibold text-stone-600 dark:text-stone-300 shrink-0">
            Select Biblical Passage:
          </label>
          <select
            value={selectedPassageId}
            onChange={(e) => {
              setSelectedPassageId(e.target.value);
              setHighlightVerseId(null);
            }}
            className="w-full sm:w-auto px-4 py-2 rounded-xl bg-stone-50 dark:bg-stone-800 border border-stone-200 dark:border-stone-700 text-sm font-medium text-stone-900 dark:text-stone-100 focus:outline-none focus:ring-2 focus:ring-amber-600/30"
          >
            {PASSAGE_GROUPS.map((pg) => (
              <option key={pg.id} value={pg.id}>
                {pg.labelEn} · {pg.labelEs}
              </option>
            ))}
          </select>
        </div>
      </section>

      {/* Translation Methodology & Nuance Callout */}
      {nuanceNote && (
        <section className="p-5 rounded-2xl bg-amber-50/70 dark:bg-amber-950/40 border border-amber-200/80 dark:border-amber-800/60 flex items-start gap-3">
          <Info className="w-5 h-5 text-amber-800 dark:text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs sm:text-sm space-y-1 text-stone-800 dark:text-stone-200">
            <div className="font-semibold text-amber-900 dark:text-amber-300">
              Linguistic & Cross-Lingual Nuance: "{nuanceNote.termEn}" (KJV) ↔ "{nuanceNote.termEs}" (RVR1960)
            </div>
            <p className="leading-relaxed text-stone-700 dark:text-stone-300 font-sans">
              {nuanceNote.note}
            </p>
          </div>
        </section>
      )}

      {/* Synchronized Parallel Columns */}
      <section className="space-y-4">
        {/* Column Headers */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 px-2 text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
          <div className="flex items-center justify-between border-b border-stone-200 dark:border-stone-800 pb-2">
            <span>King James Version (KJV) · English (1611)</span>
            <span className="font-mono text-[10px]">Formal Equivalence</span>
          </div>
          <div className="flex items-center justify-between border-b border-stone-200 dark:border-stone-800 pb-2">
            <span>Reina-Valera 1960 (RVR1960) · Español</span>
            <span className="font-mono text-[10px] text-amber-700 dark:text-amber-400">Equivalencia Formal</span>
          </div>
        </div>

        {/* Verse Pairs */}
        <div className="space-y-3">
          {currentVerses.map((verse) => {
            const isHighlighted = highlightVerseId === verse.id;
            return (
              <div
                key={verse.id}
                className={`p-5 rounded-2xl border transition-all ${
                  isHighlighted
                    ? 'bg-amber-50/90 dark:bg-amber-950/60 border-amber-400 dark:border-amber-700 ring-2 ring-amber-400/30'
                    : 'bg-white dark:bg-stone-900 border-stone-200 dark:border-stone-800 hover:border-stone-300 dark:hover:border-stone-700'
                }`}
              >
                {/* Verse Number Indicator */}
                <div className="flex items-center justify-between text-xs text-stone-400 dark:text-stone-500 pb-2 border-b border-stone-100 dark:border-stone-800 mb-3">
                  <span className="font-mono font-bold text-amber-800 dark:text-amber-400">
                    Verse {verse.verse}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-stone-400">
                      {verse.theologicalThemes.slice(0, 2).join(' · ')}
                    </span>
                    <button
                      onClick={() =>
                        onAskQuestionAboutPassage(
                          `Explain the meaning and application of ${verse.bookNameEn} ${verse.chapter}:${verse.verse} ("${verse.kjv}").`
                        )
                      }
                      className="text-amber-700 dark:text-amber-400 hover:underline"
                    >
                      Ask AI
                    </button>
                  </div>
                </div>

                {/* Parallel Verse Text */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
                  {/* KJV Side */}
                  <div className="space-y-1">
                    <p className="font-serif text-lg text-stone-900 dark:text-stone-100 leading-relaxed">
                      <span className="font-bold text-stone-400 dark:text-stone-500 text-sm mr-2 select-none">
                        {verse.verse}
                      </span>
                      {verse.kjv}
                    </p>
                  </div>

                  {/* RVR1960 Side */}
                  <div className="space-y-1">
                    <p className="font-serif text-lg text-stone-900 dark:text-stone-100 leading-relaxed">
                      <span className="font-bold text-amber-700 dark:text-amber-400 text-sm mr-2 select-none">
                        {verse.verse}
                      </span>
                      {verse.rvr1960}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};
