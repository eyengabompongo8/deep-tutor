import React, { useState, useRef, useMemo } from 'react';
import { Link2, Lightbulb, X, Sparkles } from 'lucide-react';
import { isWordMatchingLemma } from './SentenceCard';

export default function WordAlignmentView({
  srcWords = [],
  refWords = [],
  tgtWords = [], // fallback alias for refWords
  alignedRefMap = {},
  alignedTargetsMap = {}, // fallback alias for alignedRefMap
  studentWords = [],
  alignedStudentMap = {},
  firstTimeLemmas = [],
  firstTimePairs = [],
}) {
  const effectiveRefWords = refWords.length > 0 ? refWords : tgtWords;
  const effectiveRefMap = Object.keys(alignedRefMap).length > 0 ? alignedRefMap : alignedTargetsMap;

  const [selectedSrcIndices, setSelectedSrcIndices] = useState([]);
  const [selectedRefIndices, setSelectedRefIndices] = useState([]);
  const [selectedStudentIndices, setSelectedStudentIndices] = useState([]);
  const [activeOrigin, setActiveOrigin] = useState('src'); // 'src', 'ref', 'student'

  const srcContainerRef = useRef(null);
  const refContainerRef = useRef(null);
  const studentContainerRef = useRef(null);

  // Precompute reverse Reference -> Source mapping
  const reverseRefMap = useMemo(() => {
    const rev = {};
    Object.entries(effectiveRefMap || {}).forEach(([s, targets]) => {
      const sInt = parseInt(s, 10);
      (targets || []).forEach(t => {
        if (!rev[t]) rev[t] = [];
        rev[t].push(sInt);
      });
    });
    return rev;
  }, [effectiveRefMap]);

  // Precompute reverse Student -> Source mapping
  const reverseStudentMap = useMemo(() => {
    const rev = {};
    Object.entries(alignedStudentMap || {}).forEach(([s, targets]) => {
      const sInt = parseInt(s, 10);
      (targets || []).forEach(t => {
        if (!rev[t]) rev[t] = [];
        rev[t].push(sInt);
      });
    });
    return rev;
  }, [alignedStudentMap]);

  // Set of target reference indices that match first-time roots
  const firstTimeTgtSet = useMemo(() => {
    if (!effectiveRefWords.length || (!firstTimeLemmas.length && !firstTimePairs.length)) return new Set();
    const set = new Set();
    effectiveRefWords.forEach((word, idx) => {
      const cleanWord = word.toLowerCase().replace(/^[^\w\u00C0-\u017F]+|[^\w\u00C0-\u017F]+$/g, '')
        .replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '');
      const isMatch = isWordMatchingLemma(word, firstTimeLemmas) ||
        firstTimePairs.some(p => {
          const pWord = (p.word || '').toLowerCase().replace(/^(se\s+|s['’]|l['’]|d['’]|j['’]|c['’]|m['’]|n['’]|t['’])/i, '');
          return pWord && pWord === cleanWord;
        });
      if (isMatch) {
        set.add(idx);
      }
    });
    return set;
  }, [effectiveRefWords, firstTimeLemmas, firstTimePairs]);

  // Set of source indices that align to first-time reference words
  const firstTimeSrcSet = useMemo(() => {
    if (!firstTimeTgtSet.size || !effectiveRefMap) return new Set();
    const set = new Set();
    Object.entries(effectiveRefMap).forEach(([s, targets]) => {
      if ((targets || []).some(t => firstTimeTgtSet.has(t))) {
        set.add(parseInt(s, 10));
      }
    });
    return set;
  }, [effectiveRefMap, firstTimeTgtSet]);

  if (!srcWords.length || (!effectiveRefWords.length && !studentWords.length)) {
    return null;
  }

  const clearSelection = () => {
    setSelectedSrcIndices([]);
    setSelectedRefIndices([]);
    setSelectedStudentIndices([]);
  };

  const getIntersectedIndices = (container, selector, attr) => {
    const selection = window.getSelection();
    if (!selection || !container || selection.isCollapsed || !selection.toString().trim()) {
      return [];
    }
    if (!container.contains(selection.anchorNode) && !container.contains(selection.focusNode)) {
      return [];
    }
    if (selection.rangeCount === 0) return [];
    const range = selection.getRangeAt(0);
    const spans = container.querySelectorAll(selector);
    const matched = [];

    spans.forEach(span => {
      if (range.intersectsNode(span)) {
        const spanRange = document.createRange();
        spanRange.selectNodeContents(span);
        const startEnd = range.compareBoundaryPoints(Range.START_TO_END, spanRange);
        const endStart = range.compareBoundaryPoints(Range.END_TO_START, spanRange);
        if (startEnd > 0 && endStart < 0) {
          const idx = parseInt(span.getAttribute(attr), 10);
          if (!isNaN(idx)) matched.push(idx);
        }
      }
    });
    return matched;
  };

  // Selection in Spanish Source
  const handleSrcSelection = () => {
    const matched = getIntersectedIndices(srcContainerRef.current, '[data-src-idx]', 'data-src-idx');
    if (matched.length === 0) {
      if (srcContainerRef.current?.contains(window.getSelection()?.anchorNode)) clearSelection();
      return;
    }
    setSelectedSrcIndices(matched);
    setActiveOrigin('src');

    // Compute aligned reference indices
    const rSet = new Set();
    matched.forEach(s => (effectiveRefMap[s] || []).forEach(r => rSet.add(r)));
    setSelectedRefIndices(Array.from(rSet).sort((a, b) => a - b));

    // Compute aligned student indices
    const stSet = new Set();
    matched.forEach(s => (alignedStudentMap[s] || []).forEach(st => stSet.add(st)));
    setSelectedStudentIndices(Array.from(stSet).sort((a, b) => a - b));
  };

  // Selection in Reference
  const handleRefSelection = () => {
    const matched = getIntersectedIndices(refContainerRef.current, '[data-ref-idx]', 'data-ref-idx');
    if (matched.length === 0) {
      if (refContainerRef.current?.contains(window.getSelection()?.anchorNode)) clearSelection();
      return;
    }
    setSelectedRefIndices(matched);
    setActiveOrigin('ref');

    // Compute source indices
    const sSet = new Set();
    matched.forEach(r => (reverseRefMap[r] || []).forEach(s => sSet.add(s)));
    const srcList = Array.from(sSet).sort((a, b) => a - b);
    setSelectedSrcIndices(srcList);

    // Compute student indices from source
    const stSet = new Set();
    srcList.forEach(s => (alignedStudentMap[s] || []).forEach(st => stSet.add(st)));
    setSelectedStudentIndices(Array.from(stSet).sort((a, b) => a - b));
  };

  // Selection in Student translation
  const handleStudentSelection = () => {
    const matched = getIntersectedIndices(studentContainerRef.current, '[data-student-idx]', 'data-student-idx');
    if (matched.length === 0) {
      if (studentContainerRef.current?.contains(window.getSelection()?.anchorNode)) clearSelection();
      return;
    }
    setSelectedStudentIndices(matched);
    setActiveOrigin('student');

    // Compute source indices
    const sSet = new Set();
    matched.forEach(st => (reverseStudentMap[st] || []).forEach(s => sSet.add(s)));
    const srcList = Array.from(sSet).sort((a, b) => a - b);
    setSelectedSrcIndices(srcList);

    // Compute reference indices from source
    const rSet = new Set();
    srcList.forEach(s => (effectiveRefMap[s] || []).forEach(r => rSet.add(r)));
    setSelectedRefIndices(Array.from(rSet).sort((a, b) => a - b));
  };

  const hasSelection = selectedSrcIndices.length > 0 || selectedRefIndices.length > 0 || selectedStudentIndices.length > 0;
  const selectedSrcWords = selectedSrcIndices.map(i => srcWords[i]).filter(Boolean).join(' ');
  const selectedRefWords = selectedRefIndices.map(i => effectiveRefWords[i]).filter(Boolean).join(' ');
  const selectedStudentWords = selectedStudentIndices.map(i => studentWords[i]).filter(Boolean).join(' ');

  return (
    <div className="p-3.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/60 border border-neutral-200/50 dark:border-neutral-700/60 space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-1.5 text-xs font-medium text-neutral-700 dark:text-neutral-300">
          <Link2 className="w-3.5 h-3.5 text-neutral-500 dark:text-neutral-400" />
          <span>Word Alignment Mapping</span>
        </div>
        <div className="text-[11px] text-neutral-400 dark:text-neutral-500 flex items-center gap-1">
          <Lightbulb className="w-3 h-3 text-amber-500" />
          <span>Highlight words in any sentence to inspect alignment</span>
        </div>
      </div>

      {/* 1. Spanish Source */}
      <div className="space-y-1">
        <div className="text-[10px] font-medium text-neutral-400 dark:text-neutral-500 uppercase tracking-wider flex items-center justify-between">
          <span>🇪🇸 Spanish</span>
          {selectedSrcIndices.length > 0 && (
            <span className="text-[10px] text-neutral-500 dark:text-neutral-400 lowercase">
              ({activeOrigin === 'src' ? 'selected' : 'aligned'})
            </span>
          )}
        </div>
        <div
          ref={srcContainerRef}
          onMouseUp={handleSrcSelection}
          onTouchEnd={handleSrcSelection}
          onKeyUp={handleSrcSelection}
          className="p-2.5 rounded-lg bg-white dark:bg-neutral-900 border border-neutral-200/50 dark:border-neutral-750 text-xs leading-relaxed text-neutral-800 dark:text-neutral-200 select-text selection:bg-amber-200 dark:selection:bg-amber-800 selection:text-neutral-900 dark:selection:text-neutral-100 cursor-text"
        >
          {srcWords.map((word, idx) => {
            const isSelected = selectedSrcIndices.includes(idx);
            const isFirstTime = !isSelected && firstTimeSrcSet.has(idx);
            return (
              <span
                key={idx}
                data-src-idx={idx}
                className={`inline transition-colors duration-150 rounded px-1 py-0.5 ${
                  isSelected
                    ? activeOrigin === 'src'
                      ? 'bg-neutral-200 dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100 font-medium'
                      : 'bg-amber-100 dark:bg-amber-950/80 text-amber-950 dark:text-amber-200 font-semibold'
                    : isFirstTime
                      ? 'bg-purple-100 dark:bg-purple-950/80 text-purple-900 dark:text-purple-200 font-semibold border-b-2 border-purple-400 dark:border-purple-500'
                      : ''
                }`}
                title={isFirstTime ? `Nouveau mot (${firstTimeLemmas.join(', ')})` : undefined}
              >
                {word}{idx < srcWords.length - 1 ? ' ' : ''}
              </span>
            );
          })}
        </div>
      </div>

      {/* 2. French Reference (Target) */}
      {effectiveRefWords.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] font-medium text-neutral-400 dark:text-neutral-500 uppercase tracking-wider flex items-center justify-between">
            <span>🇫🇷 Reference (Target)</span>
            {selectedRefIndices.length > 0 && (
              <span className="text-[10px] text-neutral-500 dark:text-neutral-400 lowercase">
                ({activeOrigin === 'ref' ? 'selected' : 'aligned'})
              </span>
            )}
          </div>
          <div
            ref={refContainerRef}
            onMouseUp={handleRefSelection}
            onTouchEnd={handleRefSelection}
            onKeyUp={handleRefSelection}
            className="p-2.5 rounded-lg bg-white dark:bg-neutral-900 border border-neutral-200/50 dark:border-neutral-750 text-xs leading-relaxed text-neutral-800 dark:text-neutral-200 select-text selection:bg-amber-200 dark:selection:bg-amber-800 selection:text-neutral-900 dark:selection:text-neutral-100 cursor-text"
          >
            {effectiveRefWords.map((word, idx) => {
              const isSelected = selectedRefIndices.includes(idx);
              const isFirstTime = !isSelected && firstTimeTgtSet.has(idx);
              return (
                <span
                  key={idx}
                  data-ref-idx={idx}
                  className={`inline transition-colors duration-150 rounded px-1 py-0.5 ${
                    isSelected
                      ? activeOrigin === 'ref'
                        ? 'bg-neutral-200 dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100 font-medium'
                        : 'bg-amber-100 dark:bg-amber-950/80 text-amber-950 dark:text-amber-200 font-semibold'
                      : isFirstTime
                        ? 'bg-purple-100 dark:bg-purple-950/80 text-purple-900 dark:text-purple-200 font-semibold border border-purple-300 dark:border-purple-800 shadow-xs'
                        : ''
                  }`}
                  title={isFirstTime ? `Nouveau mot (${firstTimeLemmas.join(', ')})` : undefined}
                >
                  {word}{idx < effectiveRefWords.length - 1 ? ' ' : ''}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* 3. Your Translation (Student) */}
      {studentWords.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] font-medium text-neutral-400 dark:text-neutral-500 uppercase tracking-wider flex items-center justify-between">
            <span>✍️ Your Translation</span>
            {selectedStudentIndices.length > 0 && (
              <span className="text-[10px] text-neutral-500 dark:text-neutral-400 lowercase">
                ({activeOrigin === 'student' ? 'selected' : 'aligned'})
              </span>
            )}
          </div>
          <div
            ref={studentContainerRef}
            onMouseUp={handleStudentSelection}
            onTouchEnd={handleStudentSelection}
            onKeyUp={handleStudentSelection}
            className="p-2.5 rounded-lg bg-white dark:bg-neutral-900 border border-neutral-200/50 dark:border-neutral-750 text-xs leading-relaxed text-neutral-800 dark:text-neutral-200 select-text selection:bg-amber-200 dark:selection:bg-amber-800 selection:text-neutral-900 dark:selection:text-neutral-100 cursor-text"
          >
            {studentWords.map((word, idx) => {
              const isSelected = selectedStudentIndices.includes(idx);
              return (
                <span
                  key={idx}
                  data-student-idx={idx}
                  className={`inline transition-colors duration-150 rounded px-0.5 ${
                    isSelected
                      ? activeOrigin === 'student'
                        ? 'bg-neutral-200 dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100 font-medium'
                        : 'bg-amber-100 dark:bg-amber-950/80 text-amber-950 dark:text-amber-200 font-semibold'
                      : ''
                  }`}
                >
                  {word}{idx < studentWords.length - 1 ? ' ' : ''}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Alignment Mapping Breakdown Banner */}
      {hasSelection && (
        <div className="p-2.5 rounded-lg bg-amber-50 dark:bg-amber-950/40 border border-amber-200/60 dark:border-amber-900/50 text-neutral-800 dark:text-neutral-200 flex items-center justify-between gap-3 text-xs animate-in fade-in duration-150">
          <div className="flex items-center gap-2">
            <Link2 className="w-3.5 h-3.5 text-amber-700 dark:text-amber-400 shrink-0" />
            <div className="space-y-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-neutral-500 dark:text-neutral-400 text-[11px]">Spanish:</span>
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">{selectedSrcWords || '—'}</span>
                <span className="text-neutral-400 dark:text-neutral-600">·</span>
                <span className="text-neutral-500 dark:text-neutral-400 text-[11px]">Reference:</span>
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">{selectedRefWords || '—'}</span>
                {studentWords.length > 0 && (
                  <>
                    <span className="text-neutral-400 dark:text-neutral-600">·</span>
                    <span className="text-neutral-500 dark:text-neutral-400 text-[11px]">You:</span>
                    <span className="font-semibold text-neutral-900 dark:text-neutral-100">{selectedStudentWords || '—'}</span>
                  </>
                )}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={clearSelection}
            className="p-1 rounded text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 transition-colors"
            title="Clear selection"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}
