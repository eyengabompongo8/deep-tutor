import React from 'react';
import { Flame, Settings, Languages, Headphones } from 'lucide-react';
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
  onSwitchMode,
  onOpenSettings,
  onSwitchUser,
  onLogout,
  onUserUpdated,
}) {
  const modeData = profile?.modes?.[currentMode] || profile;
  const score = modeData?.score ?? 1.20;
  const level = modeData?.level ?? 'A1';
  const streak = modeData?.streak ?? 0;

  const currentLevelInt = Math.floor(score);
  const progressPercent = Math.min(100, Math.max(0, Math.round((score - currentLevelInt) * 100)));

  const sessionCompleted = sessionProgress?.completed ?? 0;
  const sessionTotal = sessionProgress?.total ?? 0;

  return (
    <header className="w-full bg-[#fafafa] py-3.5 border-b border-neutral-200/60 sticky top-0 z-30 backdrop-blur-md bg-[#fafafa]/90">
      <div className="max-w-[1600px] w-full mx-auto px-4 sm:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4">
        {/* Score, Level, & Progress */}
        <div className="flex items-center gap-4 sm:gap-6 w-full sm:w-auto justify-between sm:justify-start">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold text-neutral-900 tracking-tight">
              {level}
            </span>
            <span className="text-sm font-mono text-neutral-500">
              {score.toFixed(2)}
            </span>
            <span className="text-xs text-neutral-400 hidden md:inline">
              · {currentMode === 'transcription' ? 'Écoute' : 'Traduction'} ({LEVEL_NAMES[level] || 'Level'})
            </span>
          </div>

          <div className="w-20 sm:w-28 bg-neutral-200 rounded-full h-1 overflow-hidden" title={`Niveau ${level} : ${progressPercent}%`}>
            <div 
              className="bg-neutral-800 h-full rounded-full transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Center: Mode Switcher */}
        <div className="flex items-center bg-neutral-200/70 p-1 rounded-xl gap-1 text-xs">
          <button
            type="button"
            onClick={() => onSwitchMode && onSwitchMode('translation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              currentMode === 'translation'
                ? 'bg-white text-neutral-900 shadow-sm font-semibold'
                : 'text-neutral-500 hover:text-neutral-900'
            }`}
          >
            <Languages className="w-3.5 h-3.5" />
            <span>Traduction</span>
          </button>
          <button
            type="button"
            onClick={() => onSwitchMode && onSwitchMode('transcription')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
              currentMode === 'transcription'
                ? 'bg-white text-neutral-900 shadow-sm font-semibold'
                : 'text-neutral-500 hover:text-neutral-900'
            }`}
          >
            <Headphones className="w-3.5 h-3.5" />
            <span>Transcription</span>
          </button>
        </div>

        {/* Right Stats, Active User & Settings */}
        <div className="flex items-center gap-3 sm:gap-4 text-xs text-neutral-500">
          {sessionTotal > 1 && (
            <div className="flex items-center gap-2.5">
              <span className="text-xs text-neutral-400 font-medium whitespace-nowrap">
                Progress <span className="text-neutral-700 font-mono font-semibold">{sessionCompleted}</span> of {sessionTotal}
              </span>
              <div className="w-16 sm:w-24 bg-neutral-200 rounded-full h-1 overflow-hidden">
                <div
                  className="bg-neutral-800 h-full rounded-full transition-all duration-300"
                  style={{ width: `${sessionTotal > 0 ? (sessionCompleted / sessionTotal) * 100 : 0}%` }}
                />
              </div>
            </div>
          )}

          {streak > 0 && (
            <div className="flex items-center gap-1 font-medium text-amber-700" title={`${streak} séries réussies`}>
              <Flame className="w-3.5 h-3.5 text-amber-500 fill-amber-500" />
              <span>{streak}</span>
            </div>
          )}

          <div className="h-4 w-px bg-neutral-200" />

          {/* Active User Icon & Selector Pill */}
          <UserMenu
            profile={profile}
            currentMode={currentMode}
            onSwitchUser={onSwitchUser}
            onLogout={onLogout}
            onUserUpdated={onUserUpdated}
          />

          <button
            onClick={onOpenSettings}
            className="p-1.5 rounded-lg text-neutral-400 hover:text-neutral-900 hover:bg-neutral-100 transition-colors"
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
