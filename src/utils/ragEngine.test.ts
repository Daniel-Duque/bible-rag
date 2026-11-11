import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cosineSimilarity,
  tokenize,
  extractQueryVector,
  bm25Score,
  hybridSearch,
  chunkPassage,
  estimateTokens,
  getCorpusStats,
} from './ragEngine';
import { BIBLE_CORPUS } from '../data/bibleCorpus';

test('cosineSimilarity calculates identical vectors as 1.0', () => {
  const v1 = [0.6, 0.8];
  const v2 = [0.6, 0.8];
  const sim = cosineSimilarity(v1, v2);
  assert.ok(Math.abs(sim - 1.0) < 0.0001);
});

test('cosineSimilarity calculates orthogonal vectors as 0.0', () => {
  const v1 = [1, 0, 0];
  const v2 = [0, 1, 0];
  const sim = cosineSimilarity(v1, v2);
  assert.equal(sim, 0);
});

test('tokenize normalizes Spanish diacritics and lowercases words', () => {
  const tokens = tokenize('¿Quién es el Pastor que consolará mi corazón?');
  assert.ok(tokens.includes('quien'));
  assert.ok(tokens.includes('pastor'));
  assert.ok(tokens.includes('consolara'));
  assert.ok(tokens.includes('corazon'));
  assert.ok(!tokens.includes('¿quién'));
});

test('extractQueryVector creates normalized unit vector for cross-lingual query', () => {
  const vecEn = extractQueryVector('The Lord is my shepherd peace in anxiety');
  const magEn = Math.sqrt(vecEn.reduce((acc, x) => acc + x * x, 0));
  assert.ok(Math.abs(magEn - 1.0) < 0.01, 'English vector should be unit length');

  const vecEs = extractQueryVector('Jehová es mi pastor paz en el afán y ansiedad');
  const magEs = Math.sqrt(vecEs.reduce((acc, x) => acc + x * x, 0));
  assert.ok(Math.abs(magEs - 1.0) < 0.01, 'Spanish vector should be unit length');

  // Both should have high similarity on the Shepherd & Peace dimensions (dims 10 and 15)
  const sim = cosineSimilarity(vecEn, vecEs);
  assert.ok(sim > 0.75, `Cross-lingual query vectors should share high similarity: got ${sim}`);
});

test('hybridSearch retrieves Psalm 23 for query "The Lord is my shepherd"', () => {
  const results = hybridSearch('The Lord is my shepherd', BIBLE_CORPUS, { mode: 'hybrid' });
  assert.ok(results.length > 0);
  assert.equal(results[0].verse.id, 'PSA.23.1');
  assert.ok(results[0].score > 0.6);
  assert.equal(results[0].verse.rvr1960, 'Jehová es mi pastor; nada me faltará.');
});

test('hybridSearch in Spanish retrieves John 3:16 for "amor de Dios y vida eterna"', () => {
  const results = hybridSearch('amor de Dios y vida eterna', BIBLE_CORPUS, { mode: 'hybrid' });
  assert.ok(results.length > 0);
  const topIds = results.slice(0, 3).map((r) => r.verse.id);
  assert.ok(topIds.includes('JHN.3.16'), 'Top 3 results should include John 3:16');
});

test('chunkPassage supports verse, pericope, and sliding_window strategies', () => {
  const sampleVerses = BIBLE_CORPUS.filter((v) => v.bookId === 'PSA' && v.chapter === 23);
  assert.equal(sampleVerses.length, 6);

  // 1. Verse level: 6 chunks
  const verseChunks = chunkPassage(sampleVerses, 'verse');
  assert.equal(verseChunks.length, 6);
  assert.equal(verseChunks[0].strategy, 'verse');

  // 2. Pericope level: 1 single thematic chunk for Psalm 23
  const pericopeChunks = chunkPassage(sampleVerses, 'pericope');
  assert.equal(pericopeChunks.length, 1);
  assert.equal(pericopeChunks[0].strategy, 'pericope');
  assert.equal(pericopeChunks[0].verseCount, 6);

  // 3. Sliding window: groups of 3
  const windowChunks = chunkPassage(sampleVerses, 'sliding_window');
  assert.ok(windowChunks.length >= 2);
  assert.equal(windowChunks[0].strategy, 'sliding_window');
  assert.equal(windowChunks[0].verseCount, 3);
});

test('estimateTokens calculates reasonable token approximations', () => {
  const tokens = estimateTokens('For God so loved the world that he gave his only begotten Son');
  assert.ok(tokens >= 14 && tokens <= 20);
});

test('getCorpusStats correctly summarizes the dataset', () => {
  const stats = getCorpusStats(BIBLE_CORPUS);
  assert.ok(stats.totalVerses > 20);
  assert.ok(stats.totalBooks >= 8);
  assert.ok(stats.oldTestamentCount > 0);
  assert.ok(stats.newTestamentCount > 0);
  assert.equal(stats.vectorDimensions, 16);
});
