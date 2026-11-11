import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import { BIBLE_CORPUS } from './src/data/bibleCorpus';
import { hybridSearch, getCorpusStats, extractQueryVector } from './src/utils/ragEngine';
import { BibleVerse, TheologicalQAResponse } from './src/types/bible';

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const isProd = process.env.NODE_ENV === 'production';

app.use(express.json());

// Initialize GoogleGenAI client server-side
const apiKey = process.env.GEMINI_API_KEY;
let ai: GoogleGenAI | null = null;
if (apiKey && apiKey !== 'MY_GEMINI_API_KEY') {
  ai = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// GET /api/stats
app.get('/api/stats', (_req: Request, res: Response) => {
  const stats = getCorpusStats(BIBLE_CORPUS);
  res.json({
    ...stats,
    hasGeminiKey: Boolean(ai),
    supportedTranslations: ['KJV (King James Version)', 'RVR1960 (Reina-Valera 1960)'],
  });
});

// GET /api/verses
app.get('/api/verses', (req: Request, res: Response) => {
  const { book, testament } = req.query;
  let results = BIBLE_CORPUS;
  if (testament && (testament === 'OT' || testament === 'NT')) {
    results = results.filter((v) => v.testament === testament);
  }
  if (book && typeof book === 'string') {
    results = results.filter((v) => v.bookId.toLowerCase() === book.toLowerCase());
  }
  res.json(results);
});

// POST /api/search
app.post('/api/search', async (req: Request, res: Response) => {
  try {
    const { query = '', mode = 'hybrid', testament = 'ALL', minScore = 0.05, topK = 15 } = req.body;
    if (!query.trim()) {
      return res.status(400).json({ error: 'Query is required' });
    }

    const results = hybridSearch(query, BIBLE_CORPUS, {
      mode,
      testament,
      minScore: Number(minScore),
      topK: Number(topK),
    });

    const queryVec = extractQueryVector(query);

    return res.json({
      query,
      resultsCount: results.length,
      queryVectorPreview: queryVec.slice(0, 8),
      results,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Search failed';
    return res.status(500).json({ error: message });
  }
});

// POST /api/ask
app.post('/api/ask', async (req: Request, res: Response) => {
  try {
    const { question = '', preferredLanguage = 'auto' } = req.body;
    if (!question.trim()) {
      return res.status(400).json({ error: 'Question is required' });
    }

    // Step 1: Semantic retrieval of relevant verses from both KJV and RVR1960
    const retrieved = hybridSearch(question, BIBLE_CORPUS, {
      mode: 'hybrid',
      topK: 6,
      minScore: 0.1,
    });

    const contextVerses = retrieved.map((r) => r.verse);

    // If Gemini is available, call gemini-3.8-flash for rich, grounded answer
    if (ai) {
      try {
        const contextStr = contextVerses
          .map(
            (v) => `[${v.bookNameEn} ${v.chapter}:${v.verse} / ${v.bookNameEs} ${v.chapter}:${v.verse}]
KJV: "${v.kjv}"
RVR1960: "${v.rvr1960}"
Theological Themes: ${v.theologicalThemes.join(', ')}`
          )
          .join('\n\n');

        const prompt = `You are a biblical scholar and theological teacher in the style of GotQuestions.org, known for providing clear, compassionate, biblically grounded answers with dual-translation citations.
The user is asking: "${question}"

Here is the retrieved bilingual Scripture context from our RAG knowledge base (King James Version and Reina-Valera 1960):
${contextStr}

Please produce a comprehensive, structured response adhering to these guidelines:
1. Ground your explanation directly in the retrieved Scripture passages.
2. Quote and reference BOTH the KJV (English) and RVR1960 (Spanish) verses accurately.
3. Compare the English and Spanish phrasing where meaningful for theological clarity.
4. Keep the tone dignified, pastoral, and objective.
5. If the user asked in Spanish, provide the summary, answer, and insights in Spanish; if in English, in English.`;

        const timeoutPromise = new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Gemini API timeout')), 4500)
        );

        const response = await Promise.race([
          ai.models.generateContent({
            model: 'gemini-3.8-flash',
            contents: prompt,
            config: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  summary: {
                    type: Type.STRING,
                    description: 'A 1-2 sentence core biblical answer to the question.',
                  },
                  detailedAnswer: {
                    type: Type.STRING,
                    description: 'A well-structured theological explanation (2-3 paragraphs) unfolding the biblical context.',
                  },
                  theologicalInsights: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: '3-4 key theological observations or cross-lingual insights between KJV and RVR1960.',
                  },
                  citations: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        verseId: { type: Type.STRING },
                        reference: { type: Type.STRING },
                        kjv: { type: Type.STRING },
                        rvr1960: { type: Type.STRING },
                        contextExplanation: { type: Type.STRING },
                      },
                      required: ['verseId', 'reference', 'kjv', 'rvr1960', 'contextExplanation'],
                    },
                  },
                  keyPrinciples: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: '3 practical or spiritual takeaways for the reader.',
                  },
                  suggestedFollowUps: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: '2-3 related questions for further biblical study.',
                  },
                },
                required: ['summary', 'detailedAnswer', 'theologicalInsights', 'citations', 'keyPrinciples', 'suggestedFollowUps'],
              },
            },
          }),
          timeoutPromise,
        ]);

        const parsed = JSON.parse(response.text || '{}') as TheologicalQAResponse;
        return res.json({
          ...parsed,
          question,
          isAiGenerated: true,
        });
      } catch (geminiError) {
        console.warn('Gemini generateContent fallback triggered:', geminiError);
      }
    }

    // High-quality deterministic fallback synthesis using retrieved verses
    const isSpanish =
      preferredLanguage === 'es' ||
      /[áéíóúñ¿¡]/i.test(question) ||
      /\b(que|como|por|dios|senor|amor|gracia|fe|paz)\b/i.test(question);

    const fallbackResponse: TheologicalQAResponse = generateDeterministicQAResponse(question, contextVerses, isSpanish);
    return res.json(fallbackResponse);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Q&A processing failed';
    return res.status(500).json({ error: message });
  }
});

function generateDeterministicQAResponse(
  question: string,
  verses: BibleVerse[],
  isSpanish: boolean
): TheologicalQAResponse {
  const topVerses = verses.slice(0, 4);
  const topRef = topVerses[0] ? `${topVerses[0].bookNameEn} ${topVerses[0].chapter}:${topVerses[0].verse}` : 'Scripture';
  const themes = Array.from(new Set(topVerses.flatMap((v) => v.theologicalThemes))).slice(0, 4);

  if (isSpanish) {
    return {
      question,
      summary: `La Escritura aborda esta pregunta fundamentándose en la soberanía, la gracia y las promesas inmutables de Dios, reveladas con claridad en pasajes clave como ${topVerses.map((v) => `${v.bookNameEs} ${v.chapter}:${v.verse}`).join(', ')}.`,
      detailedAnswer: `Al examinar el testimonio bíblico conjunto de la versión Reina-Valera 1960 y la King James Version, encontramos que la verdad divina trasciende las barreras idiomáticas. La revelación divina enfatiza que ${themes.join(', ')} forman un tejido cohesivo a través del Antiguo y Nuevo Testamento.\n\nEn la tradición protestante e histórica, las Escrituras nos enseñan a descansar no en la prudencia humana, sino en la fidelidad eterna del Creador. Cuando el creyente profundiza en la Palabra, descubre que cada promesa divina está sellada por el pacto eterno.\n\nAl comparar las traducciones, la RVR1960 provee una fuerza reverente y directa ("${topVerses[0]?.rvr1960 || ''}"), mientras que la KJV preserva la majestad del inglés jacobeo ("${topVerses[0]?.kjv || ''}"). Ambas fuentes convergen en exaltar la suficiencia de Dios.`,
      theologicalInsights: [
        `Armonía bilingüe: Tanto 'faith' (KJV) como 'fe' (RVR1960) representan una confianza activa y total en el carácter divino.`,
        `Continuidad testamentaria: Las verdades del Salterio y los Evangelios muestran un Dios inmutable en Su providencia.`,
        `Precisión semántica: Ambas traducciones reflejan fielmente el sentido de los textos originales masorético y Textus Receptus.`,
      ],
      citations: topVerses.map((v) => ({
        verseId: v.id,
        reference: `${v.bookNameEs} ${v.chapter}:${v.verse} / ${v.bookNameEn} ${v.chapter}:${v.verse}`,
        kjv: v.kjv,
        rvr1960: v.rvr1960,
        contextExplanation: `En este pasaje (${v.pericopeEs}), se enfatiza ${v.theologicalThemes.join(', ')}, ofreciendo certeza doctrinal y consuelo espiritual.`,
      })),
      keyPrinciples: [
        'Anclar las convicciones en el texto bíblico contextual y no en versículos aislados.',
        'Apreciar la riqueza de la comparación interlingüística para profundizar en el significado original.',
        'Aplicar la paz y la verdad de Dios activamente en la vida cotidiana y la oración.',
      ],
      suggestedFollowUps: [
        '¿Cómo se relaciona la justificación por la fe con las obras en el Nuevo Testamento?',
        '¿Cuál es el significado del pacto davídico a la luz de los Salmos mesiánicos?',
        '¿Cómo debemos entender la soberanía de Dios frente a las pruebas terrenales?',
      ],
      isAiGenerated: false,
    };
  }

  return {
    question,
    summary: `Scripture addresses this foundational inquiry through the unshakeable sovereignty, grace, and promises of God, notably illustrated in passages such as ${topVerses.map((v) => `${v.bookNameEn} ${v.chapter}:${v.verse}`).join(', ')}.`,
    detailedAnswer: `When examining the combined biblical witness of the King James Version and the Reina-Valera 1960, divine revelation transcends linguistic borders. The inspired Scriptures consistently testify that ${themes.join(', ')} constitute an unbroken narrative arc across both Testaments.\n\nIn classical historic Christian theology, believers are directed to trust not in human understanding or transient circumstances, but in the steadfast covenant of God. Whether traversing valleys of shadow or walking in the light of the resurrection, the Word of God provides an anchor for the soul.\n\nComparing both translations enriches our understanding: the KJV delivers rhythmic majesty ("${topVerses[0]?.kjv || ''}"), while the RVR1960 delivers clear, reverent Hispanic cadence ("${topVerses[0]?.rvr1960 || ''}"). Both point to the solitary sufficiency of Christ.`,
    theologicalInsights: [
      `Cross-lingual fidelity: Both KJV and RVR1960 maintain strict lexical alignment with classical Textus Receptus and Masoretic traditions.`,
      `Theological coherence: Themes of peace, divine guidance, and redemption recur harmoniously across poetic and epistolary genres.`,
      `Contextual weight: Pericope-level reading preserves the full doctrinal intent beyond single-verse proof-texting.`,
    ],
    citations: topVerses.map((v) => ({
      verseId: v.id,
      reference: `${v.bookNameEn} ${v.chapter}:${v.verse} (${v.bookNameEs})`,
      kjv: v.kjv,
      rvr1960: v.rvr1960,
      contextExplanation: `Within the context of ${v.pericopeEn}, this passage establishes ${v.theologicalThemes.join(', ')}, assuring believers of God's abiding presence.`,
    })),
    keyPrinciples: [
      'Interpret difficult passages through the broader, clear counsel of Scripture (Scripture interprets Scripture).',
      'Utilize cross-translation comparison to uncover nuances of archaic or specialized theological vocabulary.',
      'Transform biblical knowledge into active faith, prayer, and discipleship in daily living.',
    ],
    suggestedFollowUps: [
      'What does Romans 8 teach regarding the eternal security of the believer?',
      'How do the Beatitudes in Matthew 5 define true Christian character?',
      'What is the spiritual significance of the Whole Armor of God in Ephesians 6?',
    ],
    isAiGenerated: false,
  };
}

// Dev server or static files
async function startServer() {
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static('dist'));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile('dist/index.html', { root: '.' });
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
