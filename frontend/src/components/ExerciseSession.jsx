import React, { useState, useEffect, useRef } from 'react';
import SentenceCard from './SentenceCard';
import TranscriptionCard from './TranscriptionCard';
import ScoreDeltaBanner from './ScoreDeltaBanner';

export default function ExerciseSession({
  batch = [],
  currentScore,
  mode = 'translation',
  userId,
  transcriptionRetry = true,
  translationThreshold = 75,
  blindRetryEnabled = false,
  onBatchCompleted,
  onNextBatch,
  isLoadingNext,
}) {
  const [evaluations, setEvaluations] = useState({});
  const [firstTryEvaluations, setFirstTryEvaluations] = useState({});
  const [evaluatingIds, setEvaluatingIds] = useState(new Set());
  const [batchSummary, setBatchSummary] = useState(null);
  const [isFinishingBatch, setIsFinishingBatch] = useState(false);
  const [preloadedHints, setPreloadedHints] = useState({});

  const hasCompletedBatchRef = useRef(false);
  const initialScoreRef = useRef(currentScore);

  // Store refs to sentence inputs for smooth Enter navigation
  const inputRefs = useRef({});

  const registerInput = (index, ref) => {
    if (ref) {
      inputRefs.current[index] = ref;
    }
  };

  // Reset local state when a new batch is passed in
  useEffect(() => {
    setEvaluations({});
    setFirstTryEvaluations({});
    setEvaluatingIds(new Set());
    setBatchSummary(null);
    setIsFinishingBatch(false);
    inputRefs.current = {};
    setPreloadedHints({});
    hasCompletedBatchRef.current = false;
    initialScoreRef.current = currentScore;

    // Sequentially prefetch NMT hint mappings in translation mode only
    let isCancelled = false;
    async function preloadHintsSequentially() {
      for (const sentence of batch) {
        if (isCancelled) break;
        const sourceText = sentence.es || sentence.en || '';
        if (!sourceText || !sentence.fr) continue;
        try {
          const res = await fetch('/api/exercises/hint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              sentence_id: sentence.id,
              source_text: sourceText,
              reference_text: sentence.fr,
              user_id: userId || 'default_user',
              selected_indices: [],
              first_time_lexemas: sentence.first_time_lexemas || sentence.new_lexemas || []
            }),
          });
          if (res.ok && !isCancelled) {
            const data = await res.json();
            setPreloadedHints(prev => ({
              ...prev,
              [sentence.id]: data,
            }));
          }
        } catch (err) {
          console.warn('Hint preloading warning:', err);
        }
      }
    }

    if (mode === 'translation' && batch && batch.length > 0) {
      preloadHintsSequentially();
    }

    return () => {
      isCancelled = true;
    };
  }, [batch, mode]);

  // When all sentences in the batch are finished, trigger batch completion
  useEffect(() => {
    if (!batch || batch.length === 0) return;

    const allEvaluated = batch.every(s => evaluations[s.id] !== undefined);
    const hasItems = batch.length > 0;

    if (hasItems && allEvaluated && !batchSummary && !isFinishingBatch) {
      finalizeBatch();
    }
  }, [evaluations, batch, batchSummary, isFinishingBatch]);

  // Focus the next card input
  const focusNextInput = (currentIndex) => {
    const nextIdx = currentIndex + 1;
    if (nextIdx < batch.length && inputRefs.current[nextIdx]) {
      setTimeout(() => {
        const nextElem = inputRefs.current[nextIdx];
        if (nextElem) {
          nextElem.focus();
          nextElem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 50);
    }
  };

  // Dispatches single sentence evaluation asynchronously
  const evaluateSentence = async (sentenceId, studentText, sentence, isSkipped = false) => {
    setEvaluatingIds(prev => new Set(prev).add(sentenceId));

    try {
      let result;
      if (mode === 'transcription') {
        const payload = {
          sentence_id: sentenceId,
          student_text: studentText || '',
          reference_text: sentence.fr,
          sentence_score: sentence.score_lexical ?? sentence.score ?? 1.0,
          is_skipped: isSkipped,
          user_id: userId || 'default_user',
        };

        const res = await fetch('/api/exercises/evaluate-transcription', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          throw new Error(`Transcription evaluation failed with status ${res.status}`);
        }
        result = await res.json();
      } else {
        const sourceText = sentence.es || sentence.en || '';
        const payload = {
          sentence_id: sentenceId,
          source_text: sourceText,
          student_text: studentText || '',
          reference_text: sentence.fr,
          sentence_score: sentence.score_lexical ?? sentence.score ?? 1.0,
          is_skipped: isSkipped,
          threshold: translationThreshold,
          user_id: userId || 'default_user',
        };

        const res = await fetch('/api/exercises/evaluate-single', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          throw new Error(`Evaluation failed with status ${res.status}`);
        }
        result = await res.json();
      }

      // Record first try evaluation for final user score calculation
      setFirstTryEvaluations(prev => {
        if (!prev[sentenceId]) {
          return { ...prev, [sentenceId]: result };
        }
        return prev;
      });

      // Update state for this sentence card immediately
      setEvaluations(prev => ({
        ...prev,
        [sentenceId]: result,
      }));
    } catch (err) {
      console.error('Error evaluating sentence:', err);
      // Fallback evaluation object
      setEvaluations(prev => ({
        ...prev,
        [sentenceId]: {
          sentence_id: sentenceId,
          is_skipped: isSkipped,
          quality: 0.0,
          composite_score: 0.0,
          is_french: true,
          lid_prob: 1.0,
          s_sem: 0,
          s_nmt: 0,
          s_fluency: 0,
          status_message: 'Connection error',
          syntactical_errors: [err.message || 'Evaluation request could not reach server.'],
          src_words: [],
          tgt_words: [],
          aligned_targets_map: {},
          corrected_text: studentText,
        },
      }));
    } finally {
      setEvaluatingIds(prev => {
        const next = new Set(prev);
        next.delete(sentenceId);
        return next;
      });
    }
  };

  // Validates a transcription attempt when retry is enabled
  const handleValidateTranscriptionAttempt = async (sentenceId, studentText, sentence, attemptIndex = 1) => {
    setEvaluatingIds(prev => new Set(prev).add(sentenceId));
    try {
      const payload = {
        sentence_id: sentenceId,
        student_text: studentText || '',
        reference_text: sentence.fr,
        sentence_score: sentence.score_lexical ?? sentence.score ?? 1.0,
        is_skipped: false,
        user_id: userId || 'default_user',
      };

      const res = await fetch('/api/exercises/evaluate-transcription', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Transcription evaluation failed with status ${res.status}`);
      }

      const result = await res.json();

      // Record first try evaluation strictly for final score calculation
      let firstScore = result.composite_score;
      setFirstTryEvaluations(prev => {
        if (!prev[sentenceId]) {
          return { ...prev, [sentenceId]: result };
        }
        firstScore = prev[sentenceId].composite_score;
        return prev;
      });

      // If exact or if retry is toggled off, complete sentence immediately
      if (result.is_exact || !transcriptionRetry) {
        const completedResult = {
          ...result,
          first_try_score: firstScore,
          attempts_taken: attemptIndex,
        };
        setEvaluations(prev => ({
          ...prev,
          [sentenceId]: completedResult,
        }));
        const idx = batch.findIndex(s => s.id === sentenceId);
        if (idx !== -1) {
          focusNextInput(idx);
        }
      }

      return result;
    } catch (err) {
      console.error('Error evaluating transcription attempt:', err);
      return {
        sentence_id: sentenceId,
        is_skipped: false,
        quality: 0.0,
        composite_score: 0.0,
        is_exact: false,
        status_message: 'Connection error',
        diff: [],
        notes: [err.message || 'Evaluation request failed'],
      };
    } finally {
      setEvaluatingIds(prev => {
        const next = new Set(prev);
        next.delete(sentenceId);
        return next;
      });
    }
  };

  const handleSentenceSubmit = (sentenceId, studentText, sentence, currentIndex) => {
    evaluateSentence(sentenceId, studentText, sentence, false);
    focusNextInput(currentIndex);
  };

  const handleSentenceSkip = (sentenceId, sentence, currentIndex) => {
    evaluateSentence(sentenceId, '', sentence, true);
    focusNextInput(currentIndex);
  };

  const handleSentenceRetry = (sentenceId, sentence, currentIndex) => {
    setEvaluations(prev => {
      const next = { ...prev };
      delete next[sentenceId];
      return next;
    });

    if (batchSummary) {
      setBatchSummary(null);
    }

    setTimeout(() => {
      const inputElem = inputRefs.current[currentIndex];
      if (inputElem) {
        inputElem.focus();
        inputElem.select();
      }
    }, 50);
  };

  // Calls backend to compute updated user score once all sentences are completed
  const finalizeBatch = async () => {
    setIsFinishingBatch(true);
    const isRetry = hasCompletedBatchRef.current;
    try {
      const evaluatedItems = batch.map(s => {
        // Strictly use the first try evaluation for user level score adjustment!
        const ev = firstTryEvaluations[s.id] || evaluations[s.id] || {};
        return {
          sentence_id: s.id,
          sentence_score: s.score_lexical ?? s.score ?? 1.0,
          quality: ev.quality || 0.0,
          is_skipped: ev.is_skipped ?? false,
        };
      });

      const res = await fetch('/api/exercises/complete-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_score: initialScoreRef.current ?? currentScore,
          mode: mode,
          user_id: userId || 'default_user',
          evaluated_items: evaluatedItems,
          is_retry: isRetry,
        }),
      });

      if (!res.ok) {
        throw new Error(`Complete batch failed with status ${res.status}`);
      }

      const summary = await res.json();
      hasCompletedBatchRef.current = true;
      setBatchSummary(summary);
      if (onBatchCompleted) {
        onBatchCompleted(summary);
      }
    } catch (err) {
      console.error('Error completing batch:', err);
    } finally {
      setIsFinishingBatch(false);
    }
  };

  const completedCount = Object.keys(evaluations).length;
  const totalCount = batch.length;

  return (
    <div className="space-y-6">
      {/* Session Progress Bar */}
      {totalCount > 1 && (
        <div className="flex items-center justify-between gap-3 pb-1">
          <div className="text-xs text-neutral-400 font-medium">
            Progress <span className="text-neutral-700 font-mono font-semibold">{completedCount}</span> of {totalCount}
          </div>
          <div className="w-20 sm:w-28 bg-neutral-200 rounded-full h-1 overflow-hidden">
            <div
              className="bg-neutral-800 h-full rounded-full transition-all duration-300"
              style={{ width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%` }}
            />
          </div>
        </div>
      )}

      {/* Sentence Cards List */}
      <div className="space-y-4">
        {batch.map((sentence, index) => (
          mode === 'transcription' ? (
            <TranscriptionCard
              key={sentence.id}
              sentence={sentence}
              index={index}
              total={totalCount}
              evalResult={evaluations[sentence.id]}
              isEvaluating={evaluatingIds.has(sentence.id)}
              retryEnabled={transcriptionRetry}
              onValidateAttempt={handleValidateTranscriptionAttempt}
              onSubmit={handleSentenceSubmit}
              onSkip={handleSentenceSkip}
              onRetry={handleSentenceRetry}
              registerInput={registerInput}
            />
          ) : (
            <SentenceCard
              key={sentence.id}
              sentence={sentence}
              index={index}
              total={totalCount}
              evalResult={evaluations[sentence.id]}
              isEvaluating={evaluatingIds.has(sentence.id)}
              preloadedHint={preloadedHints[sentence.id]}
              threshold={translationThreshold}
              blindRetryEnabled={blindRetryEnabled}
              onSubmit={handleSentenceSubmit}
              onSkip={handleSentenceSkip}
              onRetry={handleSentenceRetry}
              registerInput={registerInput}
            />
          )
        ))}
      </div>

      {/* Score Delta Banner (Revealed when all sentences are done and evaluated) */}
      <ScoreDeltaBanner
        batchSummary={batchSummary}
        onNextBatch={onNextBatch}
        isLoadingNext={isLoadingNext}
      />
    </div>
  );
}
