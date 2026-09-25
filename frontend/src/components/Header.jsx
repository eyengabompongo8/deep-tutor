import React from 'react';
import { Flame, Settings, Languages, Headphones, Shuffle, Sun, Moon, Monitor } from 'lucide-react';
import UserMenu from './UserMenu';

const LEVEL_NAMES = {
  A1: 'Beginner',
  A2: 'Elementary',
  B1: 'Intermediate',
  B2: 'Upper Intermediate',
  C1: 'Advanced',
  C2: 'Mastery',
};

export default function Header({
  profile,
  currentMode = 'translation',
  sessionProgress,
  theme = 'system',
  onToggleTheme,
  onSwitchMode,
  onOpenSettings,
  onSwitchUser,
  onLogout,
  onUserUpdated,
}) {
  const modeData = profile?.modes?.[currentMode] || profile?.modes?.translation || profile;
  const score = modeData?.score ?? 1.20;
  const level = modeData?.level ?? 'A1';
  const streak = modeData?.streak ?? 0;

  const currentLevelInt = Math.floor(score);
  const progressPercent = Math.min(100, Math.max(0, Math.round((score - currentLevelInt) * 100)));

  const sessionCompleted = sessionProgress?.completed ?? 0;
  const sessionTotal = sessionProgress?.total ?? 0;

  const themeLabels = {
    system: 'Thème : Système (automatique) · Cliquer pour Passer au Clair',
    light: 'Thème : Clair · Cliquer pour Passer au Sombre',
    dark: 'Thème : Sombre · Cliquer pour Passer au Système',
  };

  const getModeLabel = () => {
    if (currentMode === 'transcription') return 'Écoute';
    if (currentMode === 'mixed') return 'Mixte';
    return 'Traduction';
  };

  return (
    <header className="w-full bg-[#fafafa]/90 dark:bg-neutral-900/90 py-3.5 border-b border-neutral-200/60 dark:border-neutral-800/80 sticky top-0 z-30 backdrop-blur-md transition-colors duration-200">
      <div className="max-w-[1600px] w-full mx-auto px-4 sm:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4">
        {/* Score, Level, & Progress */}
        <div className="flex items-center gap-4 sm:gap-6 w-full sm:w-auto justify-between sm:justify-start">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 tracking-tight">
              {level}
            </span>
            <span className="text-sm font-mono text-neutral-500 dark:text-neutral-400">
              {score.toFixed(2)}
            </span>
            <span className="text-xs text-neutral-400 dark:text-neutral-500 hidden md:inline">
              · {getModeLabel()} ({LEVEL_NAMES[level] || 'Level'})
            </span>
          </div>

          <div className="w-20 sm:w-28 bg-neutral-200 dark:bg-neutral-800 rounded-full h-1 overflow-hidden" title={`Niveau ${level} : ${progressPercent}%`}>
            <div 
              className="bg-neutral-800 dark:bg-neutral-200 h-full rounded-full transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Center: Mode Switcher */}
        <div className="flex items-center bg-neutral-200/70 dark:bg-neutral-800/80 p-1 rounded-xl gap-1 text-xs">
          <button
            type="button"
            onClick={() => onSwitchMode && onSwitchMode('translation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              currentMode === 'translation'
                ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 shadow-xs font-semibold'
                : 'text-neutral-500 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
            }`}
          >
            <Languages className="w-3.5 h-3.5" />
            <span>Traduction</span>
          </button>
          <button
            type="button"
            onClick={() => onSwitchMode && onSwitchMode('mixed')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              currentMode === 'mixed'
                ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 shadow-xs font-semibold'
                : 'text-neutral-500 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
            }`}
          >
            <Shuffle className="w-3.5 h-3.5" />
            <span>Mixte</span>
          </button>
          <button
            type="button"
            onClick={() => onSwitchMode && onSwitchMode('transcription')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              currentMode === 'transcription'
                ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 shadow-xs font-semibold'
                : 'text-neutral-500 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
            }`}
          >
            <Headphones className="w-3.5 h-3.5" />
            <span>Transcription</span>
          </button>
        </div>

        {/* Right Stats, Active User, Theme Toggle & Settings */}
        <div className="flex items-center gap-3 sm:gap-4 text-xs text-neutral-500 dark:text-neutral-400">
          {sessionTotal > 1 && (
            <div className="flex items-center gap-2.5">
              <span className="text-xs text-neutral-400 dark:text-neutral-500 font-medium whitespace-nowrap">
                Progress <span className="text-neutral-700 dark:text-neutral-300 font-mono font-semibold">{sessionCompleted}</span> of {sessionTotal}
              </span>
              <div className="w-16 sm:w-24 bg-neutral-200 dark:bg-neutral-800 rounded-full h-1 overflow-hidden">
                <div
                  className="bg-neutral-800 dark:bg-neutral-200 h-full rounded-full transition-all duration-300"
                  style={{ width: `${sessionTotal > 0 ? (sessionCompleted / sessionTotal) * 100 : 0}%` }}
                />
              </div>
            </div>
          )}

          {streak > 0 && (
            <div className="flex items-center gap-1 font-medium text-amber-700 dark:text-amber-400" title={`${streak} séries réussies`}>
              <Flame className="w-3.5 h-3.5 text-amber-500 fill-amber-500" />
              <span>{streak}</span>
            </div>
          )}

          <div className="h-4 w-px bg-neutral-200 dark:bg-neutral-800" />

          {/* Active User Icon & Selector Pill */}
          <UserMenu
            profile={profile}
            currentMode={currentMode}
            onSwitchUser={onSwitchUser}
            onLogout={onLogout}
            onUserUpdated={onUserUpdated}
          />

          {/* Theme Toggle Button (Cycling System -> Light -> Dark) */}
          <button
            type="button"
            onClick={onToggleTheme}
            className="flex items-center gap-1 p-1.5 rounded-lg text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100 dark:text-neutral-400 dark:hover:text-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            title={themeLabels[theme] || themeLabels.system}
            aria-label={themeLabels[theme] || themeLabels.system}
          >
            {theme === 'system' ? (
              <Monitor className="w-4 h-4 text-neutral-500 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200 transition-colors" />
            ) : theme === 'light' ? (
              <Sun className="w-4 h-4 text-amber-500 hover:text-amber-600 transition-colors" />
            ) : (
              <Moon className="w-4 h-4 text-indigo-400 hover:text-indigo-300 transition-colors" />
            )}
          </button>

          {/* Settings Button */}
          <button
            type="button"
            onClick={onOpenSettings}
            className="p-1.5 rounded-lg text-neutral-400 hover:text-neutral-900 hover:bg-neutral-100 dark:text-neutral-400 dark:hover:text-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            title="Paramètres"
            aria-label="Paramètres"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
