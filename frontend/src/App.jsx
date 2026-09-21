import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import ExerciseSession from './components/ExerciseSession';
import SettingsModal from './components/SettingsModal';
import LoginScreen from './components/LoginScreen';
import { Loader2, AlertCircle } from 'lucide-react';

export default function App() {
  const [activeUserId, setActiveUserId] = useState(() => {
    return localStorage.getItem('frenchtutor_user_id') || null;
  });
  const [isSelectingUser, setIsSelectingUser] = useState(() => {
    return !localStorage.getItem('frenchtutor_user_id');
  });
  const [initialCreateMode, setInitialCreateMode] = useState(false);

  const [profile, setProfile] = useState(null);
  const [batch, setBatch] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingNext, setIsLoadingNext] = useState(false);
  const [currentMode, setCurrentMode] = useState(() => {
    return localStorage.getItem('frenchtutor_mode') || 'translation';
  });
  const [batchSize, setBatchSize] = useState(() => {
    const saved = localStorage.getItem('frenchtutor_batch_size');
    return saved ? parseInt(saved, 10) : 3;
  });
  const [transcriptionRetry, setTranscriptionRetry] = useState(() => {
    const saved = localStorage.getItem('frenchtutor_transcription_retry');
    return saved !== null ? saved === 'true' : true;
  });
  const [translationThreshold, setTranslationThreshold] = useState(() => {
    const saved = localStorage.getItem('frenchtutor_translation_threshold');
    return saved !== null ? parseInt(saved, 10) : 75;
  });
  const [blindRetryEnabled, setBlindRetryEnabled] = useState(() => {
    const saved = localStorage.getItem('frenchtutor_blind_retry');
    return saved !== null ? saved === 'true' : false;
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [error, setError] = useState(null);
  const [recentSentenceIds, setRecentSentenceIds] = useState([]);
  const [sessionProgress, setSessionProgress] = useState({ completed: 0, total: 0 });

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
      setError('Could not connect to the FrenchTutor server. Please ensure server.py is running.');
      return null;
    }
  }, [currentMode, activeUserId]);

  // Fetch a batch of difficulty-matched sentences
  const fetchBatch = useCallback(async (targetScore, size = batchSize, excludeIds = recentSentenceIds, mode = currentMode, uid = activeUserId) => {
    if (!uid) return;
    try {
      setError(null);
      const excludeParam = excludeIds.length > 0 ? `&exclude=${excludeIds.join(',')}` : '';
      const res = await fetch(`/api/exercises/batch?mode=${mode}&score=${targetScore}&count=${size}${excludeParam}&user_id=${encodeURIComponent(uid)}`);
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

  // Initial load when activeUserId is available
  useEffect(() => {
    if (!activeUserId) {
      setIsLoading(false);
      return;
    }
    const init = async () => {
      setIsLoading(true);
      const userProfile = await fetchProfile(currentMode, activeUserId);
      if (userProfile) {
        const score = userProfile.score ?? 1.20;
        await fetchBatch(score, batchSize, [], currentMode, activeUserId);
      }
      setIsLoading(false);
    };
    init();
  }, [activeUserId]);

  // User Selection Handler from LoginScreen
  const handleSelectUser = async (userId) => {
    setActiveUserId(userId);
    localStorage.setItem('frenchtutor_user_id', userId);
    setIsSelectingUser(false);
    setInitialCreateMode(false);
    setRecentSentenceIds([]);
    setSessionProgress({ completed: 0, total: 0 });
    setIsLoading(true);
    try {
      const userProfile = await fetchProfile(currentMode, userId);
      const score = userProfile?.score ?? 1.20;
      await fetchBatch(score, batchSize, [], currentMode, userId);
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
    localStorage.removeItem('frenchtutor_user_id');
    setActiveUserId(null);
    setProfile(null);
    setBatch([]);
    setRecentSentenceIds([]);
    setSessionProgress({ completed: 0, total: 0 });
    setInitialCreateMode(false);
    setIsSelectingUser(true);
  };

  // Profile update handler (when avatar or name is modified)
  const handleUserUpdated = (updatedUser) => {
    if (!updatedUser) return;
    setProfile(prev => {
      if (!prev) return updatedUser;
      return {
        ...prev,
        name: updatedUser.name,
        avatar: updatedUser.avatar,
      };
    });
  };

  // Mode switch handler
  const handleSwitchMode = async (newMode) => {
    if (newMode === currentMode) return;
    setCurrentMode(newMode);
    localStorage.setItem('frenchtutor_mode', newMode);
    setSessionProgress({ completed: 0, total: 0 });
    setIsLoading(true);
    try {
      const userProfile = await fetchProfile(newMode, activeUserId);
      const score = userProfile?.score ?? 1.20;
      setRecentSentenceIds([]);
      await fetchBatch(score, batchSize, [], newMode, activeUserId);
    } catch (err) {
      console.error('Failed to switch mode:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handler when a batch completes and score is updated
  const handleBatchCompleted = (summary) => {
    if (summary && summary.profile) {
      setProfile(summary.profile);
    } else if (summary && summary.new_score) {
      setProfile(prev => {
        if (!prev) return prev;
        const currentModeData = prev.modes?.[currentMode] || {};
        return {
          ...prev,
          score: summary.new_score,
          level: summary.new_level,
          modes: {
            ...prev.modes,
            [currentMode]: {
              ...currentModeData,
              score: summary.new_score,
              level: summary.new_level,
              total_completed: summary.total_completed ?? (currentModeData.total_completed || 0) + 1,
              streak: summary.streak ?? currentModeData.streak ?? 0,
            }
          }
        };
      });
    }
  };

  // Handler to load next batch with the new score
  const handleNextBatch = async () => {
    setIsLoadingNext(true);
    const scoreToUse = profile?.modes?.[currentMode]?.score ?? profile?.score ?? 1.20;
    await fetchBatch(scoreToUse, batchSize, recentSentenceIds, currentMode, activeUserId);
    setIsLoadingNext(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Change batch size from settings
  const handleChangeBatchSize = (newSize) => {
    setBatchSize(newSize);
    localStorage.setItem('frenchtutor_batch_size', String(newSize));
    const scoreToUse = profile?.modes?.[currentMode]?.score ?? profile?.score ?? 1.20;
    fetchBatch(scoreToUse, newSize, recentSentenceIds, currentMode, activeUserId);
  };

  // Reset or adjust user score from settings
  const handleResetScore = async (newScore = 1.20, modeToReset = currentMode) => {
    try {
      setIsLoading(true);
      const res = await fetch('/api/user/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_score: newScore,
          mode: modeToReset,
          user_id: activeUserId || 'default_user'
        })
      });
      const data = await res.json();
      setProfile(data);
      setRecentSentenceIds([]);
      await fetchBatch(data.score, batchSize, [], modeToReset, activeUserId);
    } catch (err) {
      console.error('Failed to reset score:', err);
    } finally {
      setIsLoading(false);
      setIsSettingsOpen(false);
    }
  };

  const handleToggleTranscriptionRetry = (val) => {
    setTranscriptionRetry(val);
    localStorage.setItem('frenchtutor_transcription_retry', String(val));
  };

  const handleToggleBlindRetry = (val) => {
    setBlindRetryEnabled(val);
    localStorage.setItem('frenchtutor_blind_retry', String(val));
  };

  const handleChangeTranslationThreshold = (val) => {
    setTranslationThreshold(val);
    localStorage.setItem('frenchtutor_translation_threshold', String(val));
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

  const activeScore = profile?.modes?.[currentMode]?.score ?? profile?.score ?? 1.20;

  return (
    <div className="min-h-screen bg-[#fafafa] text-neutral-900 flex flex-col">
      {/* Top Navigation with Active User Avatar Icon */}
      <Header
        profile={profile}
        currentMode={currentMode}
        sessionProgress={sessionProgress}
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
                fetchBatch(activeScore, batchSize, recentSentenceIds, currentMode, activeUserId);
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
            <Loader2 className="w-6 h-6 animate-spin text-neutral-600" />
            <div className="text-xs font-medium text-neutral-500">Chargement des exercices...</div>
          </div>
        ) : (
          <div>
            {/* Active Exercise Session */}
            <ExerciseSession
              batch={batch}
              currentScore={activeScore}
              mode={currentMode}
              userId={activeUserId}
              transcriptionRetry={transcriptionRetry}
              translationThreshold={translationThreshold}
              blindRetryEnabled={blindRetryEnabled}
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
        onResetScore={handleResetScore}
        transcriptionRetry={transcriptionRetry}
        onToggleTranscriptionRetry={handleToggleTranscriptionRetry}
        translationThreshold={translationThreshold}
        onChangeTranslationThreshold={handleChangeTranslationThreshold}
        blindRetryEnabled={blindRetryEnabled}
        onToggleBlindRetry={handleToggleBlindRetry}
      />

      {/* Subtle Minimalist Footer */}
      <footer className="py-8 text-center text-xs text-neutral-400">
        <div className="max-w-[1600px] w-full mx-auto px-4 sm:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Spanish → French Practice</span>
          <span>Adaptive CEFR Engine</span>
        </div>
      </footer>
    </div>
  );
}

