import React, { useEffect } from 'react';
import { ArrowRight, Trophy, TrendingUp, TrendingDown } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function ScoreDeltaBanner({
  batchSummary,
  onNextBatch,
  isLoadingNext,
}) {
  if (!batchSummary) return null;

  const { old_score, new_score, delta, old_level, new_level, level_changed, avg_quality } = batchSummary;
  const isPositive = delta >= 0;

  useEffect(() => {
    if (level_changed && isPositive) {
      confetti({
        particleCount: 60,
        spread: 60,
        origin: { y: 0.6 }
      });
    }
  }, [level_changed, isPositive]);

  return (
    <div className="p-6 sm:p-7 rounded-2xl bg-white dark:bg-neutral-900 border border-neutral-200/60 dark:border-neutral-800 shadow-[0_1px_3px_rgba(0,0,0,0.03)] text-neutral-900 dark:text-neutral-100 animate-in fade-in duration-300">
      <div className="flex flex-col md:flex-row items-center justify-between gap-5">
        {/* Left: Summary Title & Score Change */}
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-neutral-100 dark:bg-neutral-800 flex items-center justify-center text-neutral-800 dark:text-neutral-200 shrink-0">
            {level_changed ? (
              <Trophy className="w-5 h-5 text-amber-500" />
            ) : isPositive ? (
              <TrendingUp className="w-5 h-5 text-neutral-800 dark:text-neutral-200" />
            ) : (
              <TrendingDown className="w-5 h-5 text-neutral-400 dark:text-neutral-500" />
            )}
          </div>

          <div>
            <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
              {level_changed 
                ? `Level Up: Reached ${new_level}` 
                : isPositive 
                ? 'Batch Completed' 
                : 'Batch Completed'}
            </h3>

            <div className="flex items-center gap-3 mt-1 text-xs text-neutral-500 dark:text-neutral-400 flex-wrap">
              <div className="flex items-center gap-1 font-mono">
                <span>{old_score.toFixed(2)}</span>
                <span className="text-neutral-400 dark:text-neutral-600">→</span>
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">{new_score.toFixed(2)}</span>
              </div>

              <span className="font-mono text-neutral-700 dark:text-neutral-300 font-medium px-2 py-0.5 rounded bg-neutral-100 dark:bg-neutral-800">
                {isPositive ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
              </span>

              <span>
                {batchSummary.total_count !== undefined ? (
                  <>Accuracy: <strong className="text-neutral-800 dark:text-neutral-200 font-medium">{batchSummary.correct_count} / {batchSummary.total_count} correctes</strong></>
                ) : (
                  <>Accuracy: <strong className="text-neutral-800 dark:text-neutral-200 font-medium">{Math.round(avg_quality * 100)}%</strong></>
                )}
              </span>
            </div>
          </div>
        </div>

        {/* Right: Action Button */}
        <button
          onClick={onNextBatch}
          disabled={isLoadingNext}
          className="w-full md:w-auto px-5 py-2.5 rounded-xl bg-neutral-900 dark:bg-neutral-100 hover:bg-neutral-800 dark:hover:bg-neutral-200 text-white dark:text-neutral-900 font-medium text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <span>Next Sentences ({new_level})</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}
