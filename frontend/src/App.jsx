import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import ExerciseSession from './components/ExerciseSession';
import SettingsModal from './components/SettingsModal';
import LoginScreen from './components/LoginScreen';
import { Loader2, AlertCircle } from 'lucide-react';

export default function App() {
  const [activeUserId, setActiveUserId] = useState(() => {
    return localStorage.getItem('deeptutor_user_id') || localStorage.getItem('frenchtutor_user_id') || null;
  });
  const [isSelectingUser, setIsSelectingUser] = useState(() => {
    return !(localStorage.getItem('deeptutor_user_id') || localStorage.getItem('frenchtutor_user_id'));
  });
  const [initialCreateMode, setInitialCreateMode] = useState(false);

  const [profile, setProfile] = useState(null);
  const [batch, setBatch] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingNext, setIsLoadingNext] = useState(false);
  const [currentMode, setCurrentMode] = useState(() => {
    return localStorage.getItem('deeptutor_mode') || localStorage.getItem('frenchtutor_mode') || 'translation';
  });
  const [batchSize, setBatchSize] = useState(() => {
    const saved = localStorage.getItem('deeptutor_batch_size') || localStorage.getItem('frenchtutor_batch_size');
    return saved ? parseInt(saved, 10) : 3;
  });
  const [mixedTranslationRatio, setMixedTranslationRatio] = useState(() => {
    const saved = localStorage.getItem('deeptutor_mixed_translation_ratio') ?? localStorage.getItem('frenchtutor_mixed_translation_ratio');
    return saved !== null ? parseInt(saved, 10) : 50;
  });
  const [transcriptionRetry, setTranscriptionRetry] = useState(() => {
    const saved = localStorage.getItem('deeptutor_transcription_retry') ?? localStorage.getItem('frenchtutor_transcription_retry');
    return saved !== null ? saved === 'true' : true;
  });
  const [translationThreshold, setTranslationThreshold] = useState(() => {
    const saved = localStorage.getItem('deeptutor_translation_threshold') ?? localStorage.getItem('frenchtutor_translation_threshold');
    return saved !== null ? parseInt(saved, 10) : 75;
  });
  const [blindRetryEnabled, setBlindRetryEnabled] = useState(() => {
    const saved = localStorage.getItem('deeptutor_blind_retry') ?? localStorage.getItem('frenchtutor_blind_retry');
    return saved !== null ? saved === 'true' : false;
  });
  const [autoJumpEnabled, setAutoJumpEnabled] = useState(() => {
    const saved = localStorage.getItem('deeptutor_auto_jump') ?? localStorage.getItem('frenchtutor_auto_jump');
    if (saved !== null) return saved === 'true';
    const legacySaved = localStorage.getItem('deeptutor_jump_animation') ?? localStorage.getItem('frenchtutor_jump_animation');
    return legacySaved !== null ? legacySaved === 'true' : true;
  });
  const [autoPlayAudio, setAutoPlayAudio] = useState(() => {
    const saved = localStorage.getItem('deeptutor_auto_play_audio') ?? localStorage.getItem('frenchtutor_auto_play_audio');
    return saved !== null ? saved === 'true' : true;
  });
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('deeptutor_theme') || localStorage.getItem('frenchtutor_theme') || 'system';
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [error, setError] = useState(null);
  const [recentSentenceIds, setRecentSentenceIds] = useState([]);
  const [sessionProgress, setSessionProgress] = useState({ completed: 0, total: 0 });

  // Sync theme with document class and system preference changes
  useEffect(() => {
    const root = document.documentElement;
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const applyTheme = () => {
      const isDark = theme === 'dark' || (theme === 'system' && mediaQuery.matches);
      if (isDark) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    };

    applyTheme();

    const handleChange = () => {
      if (theme === 'system') {
        applyTheme();
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);

  // Fetch user profile for a specific mode and user
  const fetchProfile = useCallback(async (mode = currentMode, uid = activeUserId) => {
    if (!uid) return null;
    try {
      const res = await fetch(`/api/user/profile?mode=${mode}&user_id=${encodeURIComponent(uid)}`);
      if (!res.ok) throw new Error('Failed to load user profile');
      const data = await res.json();
      setProfile(data);
      return data;
    } catch (err) {
      console.error(err);
      setError('Could not connect to the DeepTutor server. Please ensure server.py is running.');
      return null;
    }
  }, [currentMode, activeUserId]);

  // Fetch a batch of difficulty-matched sentences
  const fetchBatch = useCallback(async (targetLevel, size = batchSize, excludeIds = recentSentenceIds, mode = currentMode, uid = activeUserId) => {
    if (!uid) return;
    try {
      setError(null);
      const lvl = targetLevel || 'A1';
      const excludeParam = excludeIds.length > 0 ? `&exclude=${excludeIds.join(',')}` : '';
      const res = await fetch(`/api/exercises/batch?mode=${mode}&level=${encodeURIComponent(lvl)}&count=${size}${excludeParam}&user_id=${encodeURIComponent(uid)}`);
      if (!res.ok) throw new Error('Failed to load exercise batch');
      const data = await res.json();
      setBatch(data.items || []);
      
      // Update recent IDs so we don't repeat them immediately
      const newIds = (data.items || []).map(s => s.id);
      setRecentSentenceIds(prev => [...prev.slice(-40), ...newIds]);
    } catch (err) {
      console.error(err);
      setError('Failed to load exercise sentences.');
    }
  }, [batchSize, recentSentenceIds, currentMode, activeUserId]);

  // Fetch user settings from backend
  const fetchSettings = useCallback(async (uid = activeUserId) => {
    if (!uid) return null;
    try {
      const res = await fetch(`/api/user/settings?user_id=${encodeURIComponent(uid)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.auto_jump !== undefined && data.auto_jump !== null) {
          setAutoJumpEnabled(Boolean(data.auto_jump));
          localStorage.setItem('deeptutor_auto_jump', String(Boolean(data.auto_jump)));
        }
        if (data.auto_play_audio !== undefined && data.auto_play_audio !== null) {
          setAutoPlayAudio(Boolean(data.auto_play_audio));
          localStorage.setItem('deeptutor_auto_play_audio', String(Boolean(data.auto_play_audio)));
        }
        if (data.cards_per_exercise != null) {
          setBatchSize(data.cards_per_exercise);
          localStorage.setItem('deeptutor_batch_size', String(data.cards_per_exercise));
        }
        if (data.mixed_translation_ratio != null) {
          setMixedTranslationRatio(data.mixed_translation_ratio);
          localStorage.setItem('deeptutor_mixed_translation_ratio', String(data.mixed_translation_ratio));
        }
        if (data.correction_threshold != null) {
          setTranslationThreshold(data.correction_threshold);
          localStorage.setItem('deeptutor_translation_threshold', String(data.correction_threshold));
        }
        if (data.transcription_retry !== undefined && data.transcription_retry !== null) {
          setTranscriptionRetry(Boolean(data.transcription_retry));
          localStorage.setItem('deeptutor_transcription_retry', String(Boolean(data.transcription_retry)));
        }
        if (data.blind_retry !== undefined && data.blind_retry !== null) {
          setBlindRetryEnabled(Boolean(data.blind_retry));
          localStorage.setItem('deeptutor_blind_retry', String(Boolean(data.blind_retry)));
        }
        if (data.theme) {
          setTheme(data.theme);
          localStorage.setItem('deeptutor_theme', data.theme);
        }
        if (data.active_mode) {
          setCurrentMode(data.active_mode);
          localStorage.setItem('deeptutor_mode', data.active_mode);
        }
        return data;
      }
      return null;
    } catch (err) {
      console.warn('Could not fetch user settings from backend:', err);
      return null;
    }
  }, [activeUserId]);

  // Initial load when activeUserId is available
  useEffect(() => {
    if (!activeUserId) {
      setIsLoading(false);
      return;
    }
    const init = async () => {
      setIsLoading(true);
      const userSettings = await fetchSettings(activeUserId);
      const modeToUse = userSettings?.active_mode || currentMode;
      const userProfile = await fetchProfile(modeToUse, activeUserId);
      if (userProfile) {
        const level = userProfile.modes?.[modeToUse]?.level ?? userProfile.level ?? 'A1';
        const effectiveBatchSize = userSettings?.cards_per_exercise ?? batchSize;
        await fetchBatch(level, effectiveBatchSize, [], modeToUse, activeUserId);
      }
      setIsLoading(false);
    };
    init();
  }, [activeUserId]);

  // User Selection Handler from LoginScreen
  const handleSelectUser = async (userId) => {
    setActiveUserId(userId);
    localStorage.setItem('deeptutor_user_id', userId);
    setIsSelectingUser(false);
    setInitialCreateMode(false);
    setRecentSentenceIds([]);
    setSessionProgress({ completed: 0, total: 0 });
    setIsLoading(true);
    try {
      const userSettings = await fetchSettings(userId);
      const modeToUse = userSettings?.active_mode || currentMode;
      const userProfile = await fetchProfile(modeToUse, userId);
      const level = userProfile?.modes?.[modeToUse]?.level ?? userProfile?.level ?? 'A1';
      const effectiveBatchSize = userSettings?.cards_per_exercise ?? batchSize;
      await fetchBatch(level, effectiveBatchSize, [], modeToUse, userId);
    } catch (err) {
      console.error('Failed to initialize selected user:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Switch User / Open Login Screen
  const handleSwitchUser = (createMode = false) => {
    setInitialCreateMode(createMode);
    setIsSelectingUser(true);
  };

  // Logout Handler
  const handleLogout = () => {
    localStorage.removeItem('deeptutor_user_id');
    localStorage.removeItem('frenchtutor_user_id');
    setActiveUserId(null);
    setProfile(null);
    setBatch([]);
    setRecentSentenceIds([]);
    setSessionProgress({ completed: 0, total: 0 });
    setInitialCreateMode(false);
    setIsSelectingUser(true);
  };

  // Profile update handler (when name is modified)
  const handleUserUpdated = (updatedUser) => {
    if (!updatedUser) return;
    setProfile(prev => {
      if (!prev) return updatedUser;
      return {
        ...prev,
        name: updatedUser.name,
      };
    });
  };

  // Mode switch handler
  const handleSwitchMode = async (newMode) => {
    if (newMode === currentMode) return;
    setCurrentMode(newMode);
    localStorage.setItem('deeptutor_mode', newMode);
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active_mode: newMode }),
      }).catch(err => console.warn('Failed to update active_mode setting:', err));
    }
    setSessionProgress({ completed: 0, total: 0 });
    setIsLoading(true);
    try {
      const userProfile = await fetchProfile(newMode, activeUserId);
      const level = userProfile?.modes?.[newMode]?.level ?? userProfile?.level ?? 'A1';
      setRecentSentenceIds([]);
      await fetchBatch(level, batchSize, [], newMode, activeUserId);
    } catch (err) {
      console.error('Failed to switch mode:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handler when a batch completes and stats are updated
  const handleBatchCompleted = (summary) => {
    if (summary && summary.profile) {
      setProfile(summary.profile);
    } else if (summary && summary.level) {
      setProfile(prev => {
        if (!prev) return prev;
        const currentModeData = prev.modes?.[currentMode] || {};
        return {
          ...prev,
          level: summary.level,
          modes: {
            ...prev.modes,
            [currentMode]: {
              ...currentModeData,
              level: summary.level,
              total_completed: summary.total_completed ?? (currentModeData.total_completed || 0) + 1,
              streak: summary.streak ?? currentModeData.streak ?? 0,
            }
          }
        };
      });
    }
  };

  // Handler to load next batch with the current level
  const handleNextBatch = async () => {
    setIsLoadingNext(true);
    const levelToUse = profile?.modes?.[currentMode]?.level ?? profile?.level ?? 'A1';
    await fetchBatch(levelToUse, batchSize, recentSentenceIds, currentMode, activeUserId);
    setIsLoadingNext(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Change batch size from settings
  const handleChangeBatchSize = (newSize) => {
    setBatchSize(newSize);
    localStorage.setItem('deeptutor_batch_size', String(newSize));
    const levelToUse = profile?.modes?.[currentMode]?.level ?? profile?.level ?? 'A1';
    fetchBatch(levelToUse, newSize, recentSentenceIds, currentMode, activeUserId);
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cards_per_exercise: newSize }),
      }).catch(err => console.warn('Failed to update cards_per_exercise:', err));
    }
  };

  // Set CEFR level from settings
  const handleSetLevel = async (newLevel = 'A1', modeToSet = currentMode) => {
    try {
      setIsLoading(true);
      const res = await fetch('/api/user/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          level: newLevel,
          mode: modeToSet,
          user_id: activeUserId || 'default_user'
        })
      });
      const data = await res.json();
      setProfile(data);
      setRecentSentenceIds([]);
      const levelToFetch = currentMode === 'mixed'
        ? undefined
        : (data.modes?.[currentMode]?.level ?? data.level ?? 'A1');
      await fetchBatch(levelToFetch, batchSize, [], currentMode, activeUserId);
    } catch (err) {
      console.error('Failed to set level:', err);
    } finally {
      setIsLoading(false);
      setIsSettingsOpen(false);
    }
  };

  const handleToggleTranscriptionRetry = (val) => {
    setTranscriptionRetry(val);
    localStorage.setItem('deeptutor_transcription_retry', String(val));
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcription_retry: val }),
      }).catch(err => console.warn('Failed to update transcription_retry setting:', err));
    }
  };

  const handleToggleBlindRetry = (val) => {
    setBlindRetryEnabled(val);
    localStorage.setItem('deeptutor_blind_retry', String(val));
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ blind_retry: val }),
      }).catch(err => console.warn('Failed to update blind_retry setting:', err));
    }
  };

  const handleChangeTranslationThreshold = (val) => {
    setTranslationThreshold(val);
    localStorage.setItem('deeptutor_translation_threshold', String(val));
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correction_threshold: val }),
      }).catch(err => console.warn('Failed to update correction_threshold:', err));
    }
  };

  const handleChangeMixedTranslationRatio = (val) => {
    setMixedTranslationRatio(val);
    localStorage.setItem('deeptutor_mixed_translation_ratio', String(val));
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mixed_translation_ratio: val }),
      }).catch(err => console.warn('Failed to update mixed_translation_ratio:', err));
    }
  };

  const handleToggleAutoJump = (val) => {
    setAutoJumpEnabled(val);
    localStorage.setItem('deeptutor_auto_jump', String(val));
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_jump: val }),
      }).catch(err => console.warn('Failed to update auto_jump setting:', err));
    }
  };

  const handleToggleAutoPlayAudio = (val) => {
    setAutoPlayAudio(val);
    localStorage.setItem('deeptutor_auto_play_audio', String(val));
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_play_audio: val }),
      }).catch(err => console.warn('Failed to update auto_play_audio setting:', err));
    }
  };

  const handleToggleTheme = (specificTheme) => {
    let nextTheme;
    if (typeof specificTheme === 'string') {
      nextTheme = specificTheme;
    } else {
      // Cycle: system -> light -> dark -> system
      if (theme === 'system') {
        nextTheme = 'light';
      } else if (theme === 'light') {
        nextTheme = 'dark';
      } else {
        nextTheme = 'system';
      }
    }
    setTheme(nextTheme);
    localStorage.setItem('deeptutor_theme', nextTheme);
    if (activeUserId) {
      fetch(`/api/user/settings?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theme: nextTheme }),
      }).catch(err => console.warn('Failed to update theme setting:', err));
    }
  };

  // If user selection / login screen is active, display it
  if (isSelectingUser || !activeUserId) {
    return (
      <LoginScreen
        activeUserId={activeUserId}
        onSelectUser={handleSelectUser}
        initialCreateMode={initialCreateMode}
      />
    );
  }

  const activeLevel = profile?.modes?.[currentMode]?.level ?? profile?.level ?? 'A1';

  return (
    <div className="min-h-screen bg-[#fafafa] dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 flex flex-col transition-colors duration-200">
      {/* Top Navigation with Active User Avatar Icon */}
      <Header
        profile={profile}
        currentMode={currentMode}
        sessionProgress={sessionProgress}
        theme={theme}
        onToggleTheme={handleToggleTheme}
        onSwitchMode={handleSwitchMode}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onSwitchUser={handleSwitchUser}
        onLogout={handleLogout}
        onUserUpdated={handleUserUpdated}
      />

      {/* Main Practice Container */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto px-4 sm:px-8 py-6 sm:py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 rounded-2xl bg-red-50 text-red-800 text-xs flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />
              <span>{error}</span>
            </div>
            <button
              onClick={() => {
                setError(null);
                fetchBatch(activeLevel, batchSize, recentSentenceIds, currentMode, activeUserId);
              }}
              className="px-3 py-1 rounded-lg bg-red-100 hover:bg-red-200 text-xs font-medium text-red-800 transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {/* Initial Loading State */}
        {isLoading ? (
          <div className="py-24 flex flex-col items-center justify-center gap-2.5 text-neutral-400">
            <Loader2 className="w-6 h-6 animate-spin text-neutral-600 dark:text-neutral-400" />
            <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400">Chargement des exercices...</div>
          </div>
        ) : (
          <div>
            {/* Active Exercise Session */}
            <ExerciseSession
              batch={batch}
              mode={currentMode}
              userId={activeUserId}
              transcriptionRetry={transcriptionRetry}
              translationThreshold={translationThreshold}
              blindRetryEnabled={blindRetryEnabled}
              autoJumpEnabled={autoJumpEnabled}
              autoPlayAudio={autoPlayAudio}
              onProgressUpdate={setSessionProgress}
              onBatchCompleted={handleBatchCompleted}
              onNextBatch={handleNextBatch}
              isLoadingNext={isLoadingNext}
            />
          </div>
        )}
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        batchSize={batchSize}
        onChangeBatchSize={handleChangeBatchSize}
        profile={profile}
        currentMode={currentMode}
        onSetLevel={handleSetLevel}
        transcriptionRetry={transcriptionRetry}
        onToggleTranscriptionRetry={handleToggleTranscriptionRetry}
        translationThreshold={translationThreshold}
        onChangeTranslationThreshold={handleChangeTranslationThreshold}
        mixedTranslationRatio={mixedTranslationRatio}
        onChangeMixedTranslationRatio={handleChangeMixedTranslationRatio}
        blindRetryEnabled={blindRetryEnabled}
        onToggleBlindRetry={handleToggleBlindRetry}
        autoJumpEnabled={autoJumpEnabled}
        onToggleAutoJump={handleToggleAutoJump}
        autoPlayAudio={autoPlayAudio}
        onToggleAutoPlayAudio={handleToggleAutoPlayAudio}
        theme={theme}
        onChangeTheme={handleToggleTheme}
      />

      {/* Subtle Minimalist Footer */}
      <footer className="py-8 text-center text-xs text-neutral-400 dark:text-neutral-500">
        <div className="max-w-[1600px] w-full mx-auto px-4 sm:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Spanish → French Practice</span>
          <span>Adaptive CEFR Engine</span>
        </div>
      </footer>
    </div>
  );
}

