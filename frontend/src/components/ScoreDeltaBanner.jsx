import React, { useEffect } from 'react';
import { ArrowRight, Trophy, Flame, CheckCircle2 } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function ScoreDeltaBanner({
  batchSummary,
  mode,
  onNextBatch,
  isLoadingNext,
}) {
  if (!batchSummary) return null;

  const { level, streak, accuracy_rate, correct_count, total_count, avg_quality } = batchSummary;
  const isPerfect = accuracy_rate === 1.0;
  const isGood = accuracy_rate >= 0.66;

  useEffect(() => {
    if (isGood) {
      confetti({
        particleCount: isPerfect ? 80 : 40,
        spread: 60,
        origin: { y: 0.6 }
      });
    }
  }, [isGood, isPerfect]);

  return (
    <div className="p-6 sm:p-7 rounded-2xl bg-white dark:bg-neutral-900 border border-neutral-200/60 dark:border-neutral-800 shadow-[0_1px_3px_rgba(0,0,0,0.03)] text-neutral-900 dark:text-neutral-100 animate-in fade-in duration-300">
      <div className="flex flex-col md:flex-row items-center justify-between gap-5">
        {/* Left: Summary Title & Stats */}
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-neutral-100 dark:bg-neutral-800 flex items-center justify-center text-neutral-800 dark:text-neutral-200 shrink-0">
            {isPerfect ? (
              <Trophy className="w-5 h-5 text-amber-500" />
            ) : (
              <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            )}
          </div>

          <div>
            <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
              <span>{isPerfect ? 'Session Parfaite !' : 'Série Terminée'}</span>
              {level && (
                <span className="px-2 py-0.5 rounded-md bg-neutral-100 dark:bg-neutral-800 text-xs font-mono font-bold text-neutral-700 dark:text-neutral-300">
                  {level}
                </span>
              )}
            </h3>

            <div className="flex items-center gap-3 mt-1 text-xs text-neutral-500 dark:text-neutral-400 flex-wrap">
              <span>
                {total_count !== undefined ? (
                  <>Précision : <strong className="text-neutral-800 dark:text-neutral-200 font-semibold">{correct_count} / {total_count} correctes</strong></>
                ) : (
                  <>Précision : <strong className="text-neutral-800 dark:text-neutral-200 font-semibold">{Math.round(avg_quality * 100)}%</strong></>
                )}
              </span>

              {streak > 0 && (
                <span className="flex items-center gap-1 font-medium text-amber-600 dark:text-amber-400">
                  <Flame className="w-3.5 h-3.5 fill-amber-500 text-amber-500" />
                  <span>{streak} en série</span>
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right: Action Button */}
        <button
          onClick={onNextBatch}
          disabled={isLoadingNext}
          className="w-full md:w-auto px-5 py-2.5 rounded-xl bg-neutral-900 dark:bg-neutral-100 hover:bg-neutral-800 dark:hover:bg-neutral-200 text-white dark:text-neutral-900 font-medium text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <span>Phrases suivantes</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}
