import React from 'react';
import { X, Sliders, RotateCcw, Moon, Sun, Monitor } from 'lucide-react';

const CEFR_PRESETS = [
  { level: 'A1', score: 1.20, label: 'Beginner' },
  { level: 'A2', score: 2.20, label: 'Elementary' },
  { level: 'B1', score: 3.20, label: 'Intermediate' },
  { level: 'B2', score: 4.20, label: 'Upper Intermediate' },
  { level: 'C1', score: 5.20, label: 'Advanced' },
];

export default function SettingsModal({
  isOpen,
  onClose,
  batchSize,
  onChangeBatchSize,
  profile,
  currentMode = 'translation',
  onResetScore,
  transcriptionRetry = true,
  onToggleTranscriptionRetry,
  translationThreshold = 75,
  onChangeTranslationThreshold,
  blindRetryEnabled = false,
  onToggleBlindRetry,
  autoJumpEnabled = true,
  onToggleAutoJump,
  theme = 'system',
  onChangeTheme,
}) {
  if (!isOpen) return null;

  const modeData = profile?.modes?.[currentMode] || profile;
  const currentScore = modeData?.score ?? 1.20;
  const currentLevel = modeData?.level ?? 'A1';
  const modeLabel = currentMode === 'transcription' ? 'Transcription' : 'Translation';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-neutral-900/40 dark:bg-black/60 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-md bg-white dark:bg-neutral-900 rounded-2xl shadow-xl p-6 sm:p-7 space-y-6 text-neutral-900 dark:text-neutral-100 border border-neutral-200/80 dark:border-neutral-800 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-1 border-b border-neutral-100 dark:border-neutral-800">
          <div className="flex items-center gap-2 font-semibold text-base text-neutral-900 dark:text-neutral-100">
            <Sliders className="w-4 h-4 text-neutral-600 dark:text-neutral-400" />
            <span>Settings</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Setting: Theme Selector */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-xs font-semibold text-neutral-800 dark:text-neutral-200 uppercase tracking-wider block">
                Thème d'affichage
              </label>
              <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                Personnalisez l'apparence de l'interface.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-1.5 pt-1">
            <button
              type="button"
              onClick={() => onChangeTheme && onChangeTheme('light')}
              className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl text-xs font-medium border transition-all ${
                theme === 'light'
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 border-transparent shadow-xs font-semibold'
                  : 'bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 border-neutral-200 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-700'
              }`}
            >
              <Sun className="w-3.5 h-3.5" />
              <span>Clair</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeTheme && onChangeTheme('dark')}
              className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl text-xs font-medium border transition-all ${
                theme === 'dark'
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 border-transparent shadow-xs font-semibold'
                  : 'bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 border-neutral-200 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-700'
              }`}
            >
              <Moon className="w-3.5 h-3.5" />
              <span>Sombre</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeTheme && onChangeTheme('system')}
              className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl text-xs font-medium border transition-all ${
                theme === 'system'
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 border-transparent shadow-xs font-semibold'
                  : 'bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 border-neutral-200 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-700'
              }`}
            >
              <Monitor className="w-3.5 h-3.5" />
              <span>Système</span>
            </button>
          </div>
        </div>

        {/* Setting 1: Sentences per screen */}
        <div className="space-y-2 pt-2 border-t border-neutral-100 dark:border-neutral-800">
          <div className="flex items-center justify-between gap-4">
            <div>
              <label className="text-xs font-semibold text-neutral-800 dark:text-neutral-200 uppercase tracking-wider block">
                Sentences per screen
              </label>
              <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                Number of sentences displayed per exercise batch (1 to 10).
              </p>
            </div>
            <div className="w-16 shrink-0">
              <input
                type="number"
                min="1"
                max="10"
                step="1"
                value={batchSize}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  if (!isNaN(val)) {
                    onChangeBatchSize(Math.max(1, Math.min(10, val)));
                  }
                }}
                className="w-full text-center py-1.5 px-2 bg-neutral-100 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-xl text-neutral-900 dark:text-neutral-100 font-mono font-semibold text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900/10 dark:focus:ring-neutral-100/10 focus:border-neutral-400 dark:focus:border-neutral-500"
              />
            </div>
          </div>
        </div>

        {/* Setting 2: Skill Level & Score Preset */}
        <div className="space-y-2 pt-2 border-t border-neutral-100 dark:border-neutral-800">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-neutral-800 dark:text-neutral-200 uppercase tracking-wider block">
              Starting Level
            </label>
            <span className="text-[11px] font-medium text-neutral-500 dark:text-neutral-400 bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 rounded-full">
              {modeLabel}
            </span>
          </div>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">
            Current proficiency: <span className="font-mono font-medium text-neutral-800 dark:text-neutral-200">{currentScore.toFixed(2)}</span> ({currentLevel})
          </p>

          <div className="pt-1">
            <select
              value={CEFR_PRESETS.some(p => p.level === currentLevel) ? currentLevel : 'A1'}
              onChange={(e) => {
                const preset = CEFR_PRESETS.find(p => p.level === e.target.value);
                if (preset) {
                  onResetScore(preset.score, currentMode);
                }
              }}
              className="w-full py-2 px-3 bg-neutral-50 dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-700 border border-neutral-200 dark:border-neutral-700 rounded-xl text-neutral-900 dark:text-neutral-100 text-xs font-medium focus:outline-none focus:ring-2 focus:ring-neutral-900/10 dark:focus:ring-neutral-100/10 focus:border-neutral-400 dark:focus:border-neutral-500 cursor-pointer transition-colors"
            >
              {CEFR_PRESETS.map((preset) => (
                <option key={preset.level} value={preset.level} className="bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100">
                  {preset.level} – {preset.label} (Score: {preset.score.toFixed(1)})
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => onResetScore(1.20, currentMode)}
            className="w-full py-2 px-3 rounded-xl bg-neutral-50 dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-700 text-xs font-medium text-neutral-600 dark:text-neutral-300 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors flex items-center justify-center gap-1.5 mt-2"
          >
            <RotateCcw className="w-3.5 h-3.5 text-neutral-400" />
            <span>Reset {modeLabel} to Beginner (A1 · 1.20)</span>
          </button>
        </div>

        {/* Setting 3: Transcription Retry Toggle */}
        <div className="space-y-2 pt-2 border-t border-neutral-100 dark:border-neutral-800">
          <div className="flex items-center justify-between gap-4">
            <div>
              <label className="text-xs font-semibold text-neutral-800 dark:text-neutral-200 uppercase tracking-wider block">
                Transcription Retry
              </label>
              <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                Mark incorrect, missing, or extra words and allow trying again.
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={transcriptionRetry}
              onClick={() => onToggleTranscriptionRetry && onToggleTranscriptionRetry(!transcriptionRetry)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                transcriptionRetry ? 'bg-neutral-900 dark:bg-neutral-100' : 'bg-neutral-200 dark:bg-neutral-700'
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white dark:bg-neutral-900 shadow ring-0 transition duration-200 ease-in-out ${
                  transcriptionRetry ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
        </div>

        {/* Setting 4: Advanced Options (Threshold & Blind Retry) */}
        <div className="space-y-4 pt-4 border-t border-neutral-100 dark:border-neutral-800">
          <div className="text-xs font-semibold text-neutral-800 dark:text-neutral-200 uppercase tracking-wider">
            Advanced Options
          </div>
          
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-4">
              <div>
                <label className="text-xs font-medium text-neutral-800 dark:text-neutral-200">
                  Translation Pass Threshold (%)
                </label>
                <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                  Minimum score required on all metrics (Semantics, NMT, Fluency) for a translation to be marked correct.
                </p>
              </div>
              <div className="w-16 shrink-0">
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="5"
                  value={translationThreshold}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    if (!isNaN(val) && onChangeTranslationThreshold) {
                      onChangeTranslationThreshold(Math.max(0, Math.min(100, val)));
                    }
                  }}
                  className="w-full text-center py-1.5 px-2 bg-neutral-100 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-xl text-neutral-900 dark:text-neutral-100 font-mono font-semibold text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900/10 dark:focus:ring-neutral-100/10 focus:border-neutral-400 dark:focus:border-neutral-500"
                />
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between gap-4 pt-2">
            <div>
              <label className="text-xs font-medium text-neutral-800 dark:text-neutral-200">
                Blind Retry (Hide Correction)
              </label>
              <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                Hide the exact reference answer when a translation is incorrect so you can try again blindly.
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={blindRetryEnabled}
              onClick={() => onToggleBlindRetry && onToggleBlindRetry(!blindRetryEnabled)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                blindRetryEnabled ? 'bg-neutral-900 dark:bg-neutral-100' : 'bg-neutral-200 dark:bg-neutral-700'
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white dark:bg-neutral-900 shadow ring-0 transition duration-200 ease-in-out ${
                  blindRetryEnabled ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between gap-4 pt-2">
            <div>
              <label className="text-xs font-medium text-neutral-800 dark:text-neutral-200">
                Auto-Jump to Next Card
              </label>
              <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                Automatically jump down with animation to the following card when a card is completed.
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={autoJumpEnabled}
              onClick={() => onToggleAutoJump && onToggleAutoJump(!autoJumpEnabled)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                autoJumpEnabled ? 'bg-neutral-900 dark:bg-neutral-100' : 'bg-neutral-200 dark:bg-neutral-700'
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white dark:bg-neutral-900 shadow ring-0 transition duration-200 ease-in-out ${
                  autoJumpEnabled ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-2">
          <button
            onClick={onClose}
            className="w-full py-2.5 px-4 rounded-xl bg-neutral-900 dark:bg-neutral-100 hover:bg-neutral-800 dark:hover:bg-neutral-200 text-white dark:text-neutral-900 text-xs font-medium transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
