import React, { useState, useRef, useEffect } from 'react';
import { CheckCircle2, AlertCircle, Sparkles, FastForward, Loader2, ArrowRight, BookOpen, AlertTriangle, Lightbulb, X, RotateCcw } from 'lucide-react';
import WordAlignmentView from './WordAlignmentView';

const FRENCH_ACCENTS = ['é', 'è', 'ê', 'ë', 'à', 'ç', 'ù', 'ô', 'î', 'œ'];

export function isWordMatchingLemma(word, lemmas = []) {
  if (!word || !lemmas || lemmas.length === 0) return false;
  let cleanWord = word.toLowerCase().replace(/^[^\w\u00C0-\u017F]+|[^\w\u00C0-\u017F]+$/g, '');
  cleanWord = cleanWord.replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '').trim();
  if (!cleanWord || cleanWord.length < 2) return false;
  return lemmas.some(lemma => {
    let cleanLemma = lemma.toLowerCase().trim();
    cleanLemma = cleanLemma.replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '').trim();
    if (cleanWord === cleanLemma) return true;
    const minLen = Math.min(cleanWord.length, cleanLemma.length);
    if (minLen >= 4 && cleanWord.slice(0, 4) === cleanLemma.slice(0, 4)) return true;
    if (minLen >= 3 && cleanWord.slice(0, 3) === cleanLemma.slice(0, 3) && Math.abs(cleanWord.length - cleanLemma.length) <= 2) return true;
    return false;
  });
}

export default function SentenceCard({
  sentence,
  index,
  total,
  evalResult,
  isEvaluating,
  preloadedHint,
  threshold = 75,
  blindRetryEnabled = false,
  onSubmit,
  onSkip,
  onRetry,
  registerInput,
}) {
  const [inputText, setInputText] = useState('');
  const [isFocused, setIsFocused] = useState(index === 0);
  const inputRef = useRef(null);
  const formRef = useRef(null);
  const sentenceTextRef = useRef(null);

  // Hints state
  const [selectedWordIndices, setSelectedWordIndices] = useState([]);
  const [hintData, setHintData] = useState(null);
  const [isLoadingHint, setIsLoadingHint] = useState(false);

  const isDone = Boolean(evalResult);

  // First-time vocabulary roots tracking
  const firstTimeLemmas = React.useMemo(() => {
    return sentence.first_time_lexemas || sentence.new_lexemas || [];
  }, [sentence]);

  // First-time vocabulary pairs (target word form + canonical lemma)
  const firstTimePairs = React.useMemo(() => {
    if (sentence.first_time_pairs && sentence.first_time_pairs.length > 0) {
      return sentence.first_time_pairs;
    }
    const lemmas = sentence.first_time_lexemas || sentence.new_lexemas || [];
    if (!lemmas.length) return [];
    const tgtTextWords = (sentence.fr || '').match(/[^\W\d_]+(?:'[^\W\d_]+)?/gu) || [];
    return lemmas.map(lem => {
      let matchedWord = lem;
      for (const w of tgtTextWords) {
        if (isWordMatchingLemma(w, [lem])) {
          matchedWord = w;
          break;
        }
      }
      return { word: matchedWord, lemma: lem };
    });
  }, [sentence]);

  // Target French word indices matching first-time roots
  const firstTimeTgtIndices = React.useMemo(() => {
    if (preloadedHint?.first_time_tgt_indices && preloadedHint.first_time_tgt_indices.length > 0) {
      return new Set(preloadedHint.first_time_tgt_indices);
    }
    if (!preloadedHint?.tgt_words || (firstTimeLemmas.length === 0 && firstTimePairs.length === 0)) return new Set();
    const set = new Set();
    preloadedHint.tgt_words.forEach((w, tIdx) => {
      const cleanW = w.toLowerCase().replace(/^[^\w\u00C0-\u017F]+|[^\w\u00C0-\u017F]+$/g, '')
        .replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '');
      const isMatch = isWordMatchingLemma(w, firstTimeLemmas) ||
        firstTimePairs.some(p => {
          const pWord = (p.word || '').toLowerCase().replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '');
          return pWord && pWord === cleanW;
        });
      if (isMatch) {
        set.add(tIdx);
      }
    });
    return set;
  }, [preloadedHint, firstTimeLemmas, firstTimePairs]);

  // Source Spanish word indices aligned to first-time roots via ML mapping
  const firstTimeSrcIndices = React.useMemo(() => {
    // 1. Direct ML-mapped source indices from preloadedHint (neural attention matrix)
    if (preloadedHint?.first_time_src_indices && preloadedHint.first_time_src_indices.length > 0) {
      return new Set(preloadedHint.first_time_src_indices);
    }
    // 2. Direct ML-mapped source indices from sentence payload if available
    if (sentence?.first_time_src_indices && sentence.first_time_src_indices.length > 0) {
      return new Set(sentence.first_time_src_indices);
    }
    // 3. Calculated using preloadedHint.aligned_targets_map and firstTimeTgtIndices
    if (preloadedHint?.aligned_targets_map && firstTimeTgtIndices.size > 0) {
      const set = new Set();
      Object.entries(preloadedHint.aligned_targets_map).forEach(([sIdx, targets]) => {
        if ((targets || []).some(t => firstTimeTgtIndices.has(t))) {
          set.add(parseInt(sIdx, 10));
        }
      });
      return set;
    }
    return new Set();
  }, [preloadedHint, sentence, firstTimeTgtIndices]);

  useEffect(() => {
    if (registerInput) {
      registerInput(index, inputRef.current);
    }
  }, [index, registerInput]);

  const sourceText = sentence.es || sentence.en || '';
  const sourceLanguageLabel = sentence.es ? 'Spanish' : (sentence.en ? 'English' : 'Source');

  // Words split from preloaded NMT words or fallback split from source sentence
  const spanishWords = preloadedHint?.src_words?.length 
    ? preloadedHint.src_words 
    : (sourceText.trim() ? sourceText.trim().split(/\s+/) : []);

  // Apply translation mapping for the given word indices
  const applyIndicesHint = (indices) => {
    if (!indices || indices.length === 0) {
      setHintData(null);
      return;
    }

    // Instant client-side lookup if hint is already preloaded
    if (preloadedHint?.aligned_targets_map && preloadedHint?.tgt_words) {
      const targetIndices = new Set();
      indices.forEach(sIdx => {
        const targets = preloadedHint.aligned_targets_map[sIdx] || preloadedHint.aligned_targets_map[String(sIdx)] || [];
        targets.forEach(t => targetIndices.add(t));
      });
      const mappedWords = Array.from(targetIndices).sort((a, b) => a - b).map(t => preloadedHint.tgt_words[t]).filter(Boolean);
      const selectedWords = indices.sort((a, b) => a - b).map(s => spanishWords[s]).filter(Boolean);
      setHintData({
        selected_source_words: selectedWords,
        mapped_target_words: mappedWords,
        hint_translation: mappedWords.join(' ')
      });
      return;
    }

    // Fallback async fetch if preloading hasn't finished yet
    setIsLoadingHint(true);
    fetch('/api/exercises/hint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sentence_id: sentence.id,
        source_text: sourceText,
        reference_text: sentence.fr,
        selected_indices: indices,
      }),
    })
    .then(res => res.ok ? res.json() : null)
    .then(data => {
      if (data) setHintData(data);
    })
    .catch(err => console.error('Failed to load hint:', err))
    .finally(() => setIsLoadingHint(false));
  };

  // Recompute hint data instantly whenever preloadedHint or selection changes
  useEffect(() => {
    if (selectedWordIndices.length > 0 && preloadedHint?.aligned_targets_map && preloadedHint?.tgt_words) {
      applyIndicesHint(selectedWordIndices);
    }
  }, [preloadedHint]);

  // Handle native text selection from the raw sentence text
  const handleTextSelection = () => {
    if (isDone) return;
    const selection = window.getSelection();
    if (!selection || !sentenceTextRef.current) return;

    const container = sentenceTextRef.current;
    if (!container.contains(selection.anchorNode) && !container.contains(selection.focusNode)) {
      return;
    }

    if (selection.isCollapsed || !selection.toString().trim()) {
      clearHint();
      return;
    }

    if (selection.rangeCount === 0) return;
    const range = selection.getRangeAt(0);

    const spans = container.querySelectorAll('[data-word-idx]');
    const matched = [];

    spans.forEach(span => {
      if (range.intersectsNode(span)) {
        const spanRange = document.createRange();
        spanRange.selectNodeContents(span);
        const startEnd = range.compareBoundaryPoints(Range.START_TO_END, spanRange);
        const endStart = range.compareBoundaryPoints(Range.END_TO_START, spanRange);
        if (startEnd > 0 && endStart < 0) {
          const idx = parseInt(span.getAttribute('data-word-idx'), 10);
          if (!isNaN(idx)) {
            matched.push(idx);
          }
        }
      }
    });

    if (matched.length > 0) {
      setSelectedWordIndices(matched);
      applyIndicesHint(matched);
    }
  };

  const clearHint = () => {
    setSelectedWordIndices([]);
    setHintData(null);
  };

  const insertAccent = (char) => {
    if (isDone || isEvaluating) return;
    const input = inputRef.current;
    if (!input) return;

    const start = input.selectionStart || inputText.length;
    const end = input.selectionEnd || inputText.length;
    const newText = inputText.substring(0, start) + char + inputText.substring(end);
    setInputText(newText);

    setTimeout(() => {
      input.focus();
      input.setSelectionRange(start + char.length, start + char.length);
    }, 10);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isDone && !isEvaluating && inputText.trim()) {
        onSubmit(sentence.id, inputText.trim(), sentence, index);
      }
    }
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    if (!isDone && !isEvaluating && inputText.trim()) {
      onSubmit(sentence.id, inputText.trim(), sentence, index);
    }
  };

  const [forceReveal, setForceReveal] = useState(false);

  const handleSkipClick = () => {
    if (!isDone && !isEvaluating) {
      setForceReveal(true);
      onSkip(sentence.id, sentence, index);
    }
  };

  const handleRevealClick = () => {
    setForceReveal(true);
  };

  const handleRetryClick = () => {
    setForceReveal(false);
    if (onRetry) {
      onRetry(sentence.id, sentence, index);
    }
  };

  const isCorrect = evalResult?.is_correct ?? false;
  const isBlindRetryActive = blindRetryEnabled && isDone && !isCorrect && !forceReveal;

  const scoreBadge = evalResult ? {
    bg: isCorrect ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' : 'bg-rose-50 text-rose-800 border border-rose-200',
    label: isCorrect ? 'Traduction correcte' : 'À corriger',
    icon: isCorrect ? CheckCircle2 : AlertCircle,
  } : null;
  const BadgeIcon = scoreBadge?.icon;
  const errorWords = (evalResult?.nmt_word_breakdown || []).filter(
    w => w.is_error && w.top_prediction && w.top_prediction !== "—" && w.top_prediction.toLowerCase() !== w.word.toLowerCase()
  );

  return (
    <div className="bg-white rounded-2xl p-6 sm:p-7 shadow-[0_1px_3px_rgba(0,0,0,0.03)] transition-all min-h-[440px] lg:h-[440px] flex flex-col justify-start">
      {/* 2-Column Side-by-Side Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-stretch flex-1 h-full min-h-0">
        
        {/* LEFT COLUMN: Sentence, Hints, & Translation Input */}
        <div className="lg:col-span-5 flex flex-col justify-between h-full min-h-0 space-y-4">
          {/* Top section: Spanish Source Sentence with Raw Text Selection */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider">
                {sourceLanguageLabel}
              </span>
              {!isDone && (
                <span className="text-[11px] text-neutral-400 flex items-center gap-1 font-normal">
                  <Lightbulb className="w-3 h-3 text-amber-500" />
                  <span>Highlight for hints</span>
                </span>
              )}
            </div>

            {/* First-time vocabulary introduction badge */}
            {firstTimePairs.length > 0 && (
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-purple-50 text-purple-700 border border-purple-200/80 text-xs font-medium mb-2.5">
                <Sparkles className="w-3.5 h-3.5 text-purple-500 shrink-0" />
                <span>
                  Nouveau vocabulaire :{' '}
                  <strong className="text-purple-900 font-semibold">
                    {firstTimePairs.map(p => {
                      const wordStr = (p.word || '').trim();
                      const lemmaStr = (p.lemma || '').trim();
                      const isIdentical = wordStr.toLowerCase() === lemmaStr.toLowerCase();
                      return isIdentical ? wordStr : `${wordStr} (${lemmaStr})`;
                    }).join(', ')}
                  </strong>
                </span>
              </div>
            )}

            {/* Raw Natural Text with Browser Selection Support */}
            <div
              ref={sentenceTextRef}
              onMouseUp={handleTextSelection}
              onTouchEnd={handleTextSelection}
              onKeyUp={handleTextSelection}
              className="text-lg sm:text-xl font-medium text-neutral-900 leading-relaxed select-text selection:bg-amber-200 selection:text-neutral-900 py-1 transition-colors cursor-text"
            >
              {spanishWords.map((word, wIdx) => {
                const isSelected = selectedWordIndices.includes(wIdx);
                const isFirstTime = firstTimeSrcIndices.has(wIdx);
                return (
                  <span
                    key={wIdx}
                    data-word-idx={wIdx}
                    className={`inline transition-colors duration-150 rounded px-1 py-0.5 ${
                      isSelected
                        ? 'bg-amber-100 text-amber-900 font-medium'
                        : isFirstTime
                          ? 'bg-purple-100 text-purple-900 font-semibold border-b-2 border-purple-400'
                          : ''
                    }`}
                    title={isFirstTime ? `Nouveau mot à apprendre (${firstTimeLemmas.join(', ')})` : undefined}
                  >
                    {word}{wIdx < spanishWords.length - 1 ? ' ' : ''}
                  </span>
                );
              })}
            </div>

            {/* Hint Translation Result */}
            {selectedWordIndices.length > 0 && (
              <div className="mt-2.5 p-3 rounded-xl bg-amber-50 text-neutral-800 flex items-center justify-between gap-3 text-xs">
                <div className="flex items-center gap-2">
                  <Lightbulb className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                  <div>
                    {isLoadingHint ? (
                      <span className="text-neutral-500">Translating...</span>
                    ) : hintData?.hint_translation ? (
                      <div>
                        <span className="text-neutral-600">"{hintData.selected_source_words?.join(' ')}"</span>
                        {' → '}
                        <strong className="text-neutral-900 font-semibold">{hintData.hint_translation}</strong>
                      </div>
                    ) : (
                      <span className="text-neutral-500">Select words to see mapping</span>
                    )}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={clearHint}
                  className="p-1 rounded text-neutral-400 hover:text-neutral-700 transition-colors"
                  title="Clear hint"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>

          {/* Interactive Translation Input or Completed View */}
          <div className="pt-2">
            {!isDone ? (
              <form ref={formRef} onSubmit={handleFormSubmit} className="space-y-3">
                <div>
                  <input
                    ref={inputRef}
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onFocus={() => setIsFocused(true)}
                    onBlur={(e) => {
                      // Only blur if not clicking accent buttons inside this form
                      if (!formRef.current?.contains(e.relatedTarget)) {
                        setIsFocused(false);
                      }
                    }}
                    onKeyDown={handleKeyDown}
                    disabled={isEvaluating}
                    placeholder="Type French translation..."
                    className="w-full px-4 py-3 bg-neutral-100 focus:bg-white text-neutral-900 placeholder-neutral-400 rounded-xl text-base focus:outline-none focus:ring-1 focus:ring-neutral-900/20 transition-all disabled:opacity-50"
                    autoFocus={index === 0}
                  />
                </div>

                {/* Accents Bar & Actions */}
                <div className="flex items-center justify-between gap-3 pt-1 min-h-[36px] flex-nowrap">
                  {/* Accent buttons: rendered with opacity when not focused to preserve height stability */}
                  <div
                    className={`flex items-center gap-1 shrink-0 transition-opacity duration-150 ${
                      isFocused ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
                    }`}
                    onMouseDown={(e) => e.preventDefault()}
                  >
                    {FRENCH_ACCENTS.map((char) => (
                      <button
                        key={char}
                        type="button"
                        onClick={() => insertAccent(char)}
                        disabled={isEvaluating}
                        className="w-7 h-7 flex items-center justify-center rounded bg-neutral-100 hover:bg-neutral-200 text-neutral-700 text-xs font-medium transition-colors disabled:opacity-50"
                      >
                        {char}
                      </button>
                    ))}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 ml-auto shrink-0">
                    <button
                      type="button"
                      onClick={handleSkipClick}
                      disabled={isEvaluating}
                      className="px-3 py-1.5 text-xs font-medium rounded-lg text-neutral-400 hover:text-neutral-800 transition-colors flex items-center gap-1 disabled:opacity-40"
                    >
                      <FastForward className="w-3.5 h-3.5" />
                      <span>Skip</span>
                    </button>

                    <button
                      type="submit"
                      disabled={!inputText.trim() || isEvaluating}
                      className="px-4 py-2 text-xs font-medium rounded-xl bg-neutral-900 hover:bg-neutral-800 text-white transition-colors flex items-center gap-1.5 disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <span>Check</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </form>
            ) : (
              /* Completed user translation */
              <div className="space-y-2">
                <div className="p-4 rounded-xl bg-neutral-50 min-h-[86px] flex flex-col justify-center">
                  <div className="text-xs text-neutral-400 font-medium mb-1">Your French Translation:</div>
                  <div className="text-base text-neutral-900 font-medium">
                    {evalResult.corrected_text || inputText || <span className="text-neutral-400 italic font-normal">—</span>}
                  </div>
                  {evalResult.autocorrect_msgs?.length > 0 && (
                    <div className="text-xs text-neutral-500 mt-2 flex items-center gap-1">
                      <span>💡 Autocorrect:</span>
                      <span className="font-mono">{evalResult.autocorrect_msgs.join(', ')}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: Evaluation Side */}
        <div className="lg:col-span-7 flex flex-col justify-start h-full min-h-0">
          
          {/* Evaluating State */}
          {isEvaluating && (
            <div className="h-full min-h-[260px] flex flex-col items-center justify-center gap-2 text-neutral-400 rounded-2xl bg-neutral-50/50 border border-neutral-100">
              <Loader2 className="w-6 h-6 animate-spin text-neutral-600" />
              <div className="text-xs font-medium text-neutral-500">Evaluating translation...</div>
            </div>
          )}

          {/* Evaluated & Completed State (Split into Sub-Left and Sub-Right) */}
          {isDone && (
            <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch h-full min-h-0">
              
              {/* SUB-LEFT: Only Reference sentence & Word Alignment with both target & student */}
              <div className="md:col-span-7 space-y-3 flex flex-col justify-start overflow-y-auto pr-1">
                {isBlindRetryActive ? (
                  <div className="p-5 rounded-2xl bg-neutral-100/50 border border-neutral-200/50 flex flex-col items-center justify-center text-center h-full min-h-[160px]">
                    <div className="w-10 h-10 rounded-full bg-white shadow-sm flex items-center justify-center text-neutral-400 mb-3">
                      <AlertCircle className="w-5 h-5" />
                    </div>
                    <div className="text-sm font-medium text-neutral-800 mb-1">
                      Correction masquée
                    </div>
                    <div className="text-xs text-neutral-500 max-w-[200px]">
                      Corrigez vos erreurs en vous aidant des scores à droite, puis réessayez.
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Reference Translation */}
                    <div className="p-3 rounded-xl bg-neutral-50 shrink-0">
                      <div className="text-xs text-neutral-500 font-medium flex items-center gap-1 mb-1">
                        <BookOpen className="w-3.5 h-3.5 text-neutral-500" />
                        <span>Reference Translation:</span>
                      </div>
                      <div className="text-base text-neutral-900 font-medium leading-relaxed">
                        {sentence.fr.split(/(\s+)/).map((part, pIdx) => {
                          const cleanPart = part.toLowerCase().replace(/^[^\w\u00C0-\u017F]+|[^\w\u00C0-\u017F]+$/g, '')
                            .replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '');
                          const isMatch = isWordMatchingLemma(part, firstTimeLemmas) ||
                            firstTimePairs.some(p => {
                              const pWord = (p.word || '').toLowerCase().replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '');
                              return pWord && pWord === cleanPart;
                            });
                          if (isMatch) {
                            return (
                              <span
                                key={pIdx}
                                className="bg-purple-100 text-purple-900 font-semibold px-1 py-0.5 rounded border border-purple-300"
                                title={`Nouveau mot (${firstTimeLemmas.join(', ')})`}
                              >
                                {part}
                              </span>
                            );
                          }
                          return <span key={pIdx}>{part}</span>;
                        })}
                      </div>
                    </div>

                    {/* Interactive Word Alignment (Showing alignment with both Target and User Translation) */}
                    {((evalResult.src_words && evalResult.src_words.length > 0) || (preloadedHint?.src_words && preloadedHint.src_words.length > 0) || spanishWords.length > 0) && (
                      <WordAlignmentView
                        srcWords={evalResult.src_words?.length ? evalResult.src_words : (preloadedHint?.src_words || spanishWords)}
                        refWords={evalResult.ref_words?.length ? evalResult.ref_words : (evalResult.tgt_words?.length ? evalResult.tgt_words : (preloadedHint?.tgt_words || []))}
                        alignedRefMap={evalResult.aligned_ref_map && Object.keys(evalResult.aligned_ref_map).length > 0 ? evalResult.aligned_ref_map : (evalResult.aligned_targets_map && Object.keys(evalResult.aligned_targets_map).length > 0 ? evalResult.aligned_targets_map : (preloadedHint?.aligned_targets_map || {}))}
                        studentWords={evalResult.student_words || []}
                        alignedStudentMap={evalResult.aligned_student_map || {}}
                        firstTimeLemmas={firstTimeLemmas}
                        firstTimePairs={firstTimePairs}
                      />
                    )}
                  </>
                )}
              </div>

              {/* SUB-RIGHT: Scores, Unexpected words, Language warning, Grammar feedback */}
              <div className="md:col-span-5 flex flex-col justify-between h-full min-h-0 space-y-3 overflow-y-auto pr-1">
                <div className="space-y-3">
                  {/* Score Badge */}
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider">
                      Evaluation
                    </span>
                    {scoreBadge && (
                      <div className={`flex items-center text-xs font-medium px-3 py-1 rounded-full ${scoreBadge.bg}`}>
                        <span>{scoreBadge.label}</span>
                      </div>
                    )}
                  </div>

                  {/* Submetrics Breakdown */}
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 rounded-xl bg-neutral-50">
                      <div className="text-[10px] text-neutral-400 font-medium">Semantic</div>
                      <div className={`text-sm font-semibold font-mono mt-0.5 ${(evalResult.s_sem ?? 0) >= threshold ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {evalResult.s_sem ?? 0}%
                      </div>
                    </div>
                    <div className="p-2 rounded-xl bg-neutral-50">
                      <div className="text-[10px] text-neutral-400 font-medium">NMT</div>
                      <div className={`text-sm font-semibold font-mono mt-0.5 ${(evalResult.s_nmt ?? 0) >= threshold ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {evalResult.s_nmt ?? 0}%
                      </div>
                    </div>
                    <div className="p-2 rounded-xl bg-neutral-50">
                      <div className="text-[10px] text-neutral-400 font-medium">Fluency</div>
                      <div className={`text-sm font-semibold font-mono mt-0.5 ${(evalResult.s_fluency ?? 0) >= threshold ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {evalResult.s_fluency ?? 0}%
                      </div>
                    </div>
                  </div>

                  {/* High Surprise Anomaly Flags (Unexpected Words) */}
                  {errorWords.length > 0 && (
                    <div className="p-2.5 rounded-xl bg-neutral-50 text-xs space-y-1 text-neutral-800">
                      <div className="font-semibold text-neutral-700 flex items-center gap-1.5">
                        <AlertCircle className="w-3.5 h-3.5 text-neutral-500" />
                        <span>Unexpected Words:</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5 pt-0.5">
                        {errorWords.map((item, i) => (
                          <span
                            key={i}
                            className="px-2 py-0.5 rounded bg-neutral-200 text-neutral-800 text-[11px]"
                            title={item.source_word ? `Expected "${item.top_prediction}" for "${item.source_word}" (ΔL: ${item.delta_l.toFixed(1)})` : `Expected: "${item.top_prediction}" (ΔL: ${item.delta_l.toFixed(1)})`}
                          >
                            "{item.word}" → <strong>{item.top_prediction}</strong>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Grammar & Syntax Error Notice */}
                  {evalResult.syntactical_errors?.length > 0 && (
                    <div className="p-3 rounded-xl bg-amber-50/90 border border-amber-200/70 text-xs shadow-xs space-y-2">
                      <div className="flex items-center gap-2 text-amber-900 font-semibold tracking-tight">
                        <div className="p-1 rounded-md bg-amber-100/80 text-amber-700">
                          <AlertCircle className="w-3.5 h-3.5" />
                        </div>
                        <span>Grammar & Syntax Feedback</span>
                        <span className="ml-auto text-[10px] font-medium px-2 py-0.5 rounded-full bg-amber-200/60 text-amber-800">
                          {evalResult.syntactical_errors.length} {evalResult.syntactical_errors.length === 1 ? 'issue' : 'issues'}
                        </span>
                      </div>
                      <div className="space-y-1.5 pt-0.5 max-h-36 overflow-y-auto">
                        {evalResult.syntactical_errors.map((err, i) => {
                          // Extract context prefix like On 'phrase': if present
                          const match = err.match(/^On '([^']+)':\s*(.+)$/);
                          if (match) {
                            const [, contextPhrase, explanation] = match;
                            return (
                              <div key={i} className="flex items-start gap-2 bg-white/70 border border-amber-100 rounded-lg p-2 text-neutral-800 leading-relaxed">
                                <span className="shrink-0 px-1.5 py-0.5 rounded bg-amber-100/90 text-amber-900 font-mono text-[11px] font-semibold">
                                  {contextPhrase}
                                </span>
                                <span className="text-neutral-700 text-[11px] pt-0.5">{explanation}</span>
                              </div>
                            );
                          }
                          return (
                            <div key={i} className="bg-white/70 border border-amber-100 rounded-lg p-2 text-neutral-700 text-[11px] leading-relaxed">
                              {err}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Language warning (only if genuine non-French) */}
                  {!evalResult.is_french && (
                    <div className="p-2.5 rounded-xl bg-red-50 text-red-800 text-xs flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-semibold">Language Warning:</span> Input was not recognized as French
                        {typeof evalResult.lid_prob === 'number' && !isNaN(evalResult.lid_prob) && evalResult.lid_prob > 0 && (
                          <span> (Confidence: {(evalResult.lid_prob * 100).toFixed(0)}%)</span>
                        )}.
                      </div>
                    </div>
                  )}
                </div>

                {/* Actions: Retry */}
                <div className="pt-2 border-t border-neutral-100 flex items-center justify-between shrink-0">
                  {evalResult.is_retry ? (
                    <div className="text-xs text-neutral-400 italic">
                      Premier essai enregistré.
                    </div>
                  ) : (
                    <div className="text-[10px] text-neutral-400">
                      Score first try only
                    </div>
                  )}
                  <div className="flex items-center gap-2 ml-auto">
                    {isBlindRetryActive && (
                      <button
                        type="button"
                        onClick={handleRevealClick}
                        className="px-3 py-1.5 text-xs font-medium rounded-xl border border-neutral-200 hover:bg-neutral-100 text-neutral-600 transition-colors"
                      >
                        Voir la réponse
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={handleRetryClick}
                      className="px-4 py-2 text-xs font-medium rounded-xl bg-neutral-900 hover:bg-neutral-800 text-white transition-colors flex items-center gap-1.5 shadow-sm"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Réessayer</span>
                    </button>
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* Idle / Waiting for Submission Placeholder */}
          {!isDone && !isEvaluating && (
            <div className="h-full p-6 text-center text-neutral-400 rounded-2xl bg-neutral-50/40 border border-neutral-100/80 flex flex-col items-center justify-center space-y-2">
              <div className="w-9 h-9 rounded-full bg-white shadow-xs border border-neutral-200/60 flex items-center justify-center text-neutral-400 mb-1">
                <Sparkles className="w-4 h-4 text-neutral-400" />
              </div>
              <div className="text-xs font-medium text-neutral-500">
                Correction & Word Mapping
              </div>
              <p className="text-[11px] text-neutral-400 max-w-xs leading-relaxed">
                Submit your translation on the left to see semantic score, neural word alignment, and grammar feedback.
              </p>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
