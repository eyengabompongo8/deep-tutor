import React, { useState, useRef, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Volume2, 
  CheckCircle2, 
  AlertCircle, 
  Sparkles, 
  FastForward, 
  Loader2, 
  ArrowRight, 
  Languages, 
  HelpCircle 
} from 'lucide-react';

const FRENCH_ACCENTS = ['é', 'è', 'ê', 'ë', 'à', 'ç', 'ù', 'ô', 'î', 'œ'];

export default function TranscriptionCard({
  sentence,
  index,
  total,
  evalResult,
  isEvaluating,
  retryEnabled = true,
  onValidateAttempt,
  onSubmit,
  onSkip,
  onRetry,
  registerInput,
}) {
  const [inputText, setInputText] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [playCount, setPlayCount] = useState(0);
  const [showTranslation, setShowTranslation] = useState(false);
  const [isFocused, setIsFocused] = useState(index === 0);

  // Retry tracking state
  const [attemptFeedback, setAttemptFeedback] = useState(null);
  const [attemptCount, setAttemptCount] = useState(1);
  const [isSubmittingAttempt, setIsSubmittingAttempt] = useState(false);

  const audioRef = useRef(null);
  const inputRef = useRef(null);

  const isDone = Boolean(evalResult);

  const audioUrl = `/api/audio/synthesize?text=${encodeURIComponent(sentence.fr)}&sentence_id=${sentence.id}`;

  useEffect(() => {
    if (registerInput) {
      registerInput(index, inputRef.current);
    }
  }, [index, registerInput, isDone]);

  // Handle audio playback speed changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackSpeed;
    }
  }, [playbackSpeed]);

  const togglePlay = (speed = playbackSpeed) => {
    if (!audioRef.current) return;
    setPlaybackSpeed(speed);
    audioRef.current.playbackRate = speed;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.currentTime = 0;
      audioRef.current.play()
        .then(() => {
          setIsPlaying(true);
          setPlayCount(prev => prev + 1);
        })
        .catch(err => console.error("Audio playback error:", err));
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const handleAccentClick = (char) => {
    if (isDone || isEvaluating) return;
    const input = inputRef.current;
    if (!input) {
      setInputText(prev => prev + char);
      return;
    }

    const start = input.selectionStart ?? inputText.length;
    const end = input.selectionEnd ?? inputText.length;
    const newText = inputText.substring(0, start) + char + inputText.substring(end);
    setInputText(newText);

    setTimeout(() => {
      input.focus();
      input.setSelectionRange(start + 1, start + 1);
    }, 10);
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (isDone || isEvaluating || isSubmittingAttempt) return;
    if (!inputText.trim()) {
      handleSkip();
      return;
    }

    if (!retryEnabled || !onValidateAttempt) {
      onSubmit(sentence.id, inputText.trim(), sentence, false);
      return;
    }

    setIsSubmittingAttempt(true);
    try {
      const result = await onValidateAttempt(sentence.id, inputText.trim(), sentence, attemptCount);
      if (result.is_exact) {
        setAttemptFeedback(null);
      } else {
        const errorItems = (result.diff || []).filter(d => d.type !== 'exact');
        setAttemptFeedback({
          ...result,
          errorCount: errorItems.length || 1,
        });
        setAttemptCount(prev => prev + 1);
        setTimeout(() => {
          if (inputRef.current) {
            inputRef.current.focus();
          }
        }, 50);
      }
    } catch (err) {
      console.error('Error validating attempt:', err);
    } finally {
      setIsSubmittingAttempt(false);
    }
  };

  const handleSkip = () => {
    if (isDone || isEvaluating || isSubmittingAttempt) return;
    setAttemptFeedback(null);
    onSkip(sentence.id, sentence, index);
  };

  const handleRetry = () => {
    if (isEvaluating || isSubmittingAttempt) return;
    setAttemptFeedback(null);
    setAttemptCount(1);
    setShowTranslation(false);
    setIsFocused(true);
    if (onRetry) {
      onRetry(sentence.id, sentence, index);
    }
  };

  // Keyboard shortcut to replay audio when focused on input
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.code === 'Space') {
      e.preventDefault();
      togglePlay(playbackSpeed);
    }
  };

  return (
    <div 
      className={`relative w-full max-w-4xl mx-auto rounded-3xl transition-colors duration-200 border border-neutral-200/60 dark:border-neutral-800 ${
        isFocused 
          ? 'bg-white dark:bg-neutral-900 shadow-[0_8px_30px_rgb(0,0,0,0.06)] ring-1 ring-neutral-900/5 dark:ring-neutral-100/5' 
          : 'bg-white/80 dark:bg-neutral-900/80 shadow-[0_2px_10px_rgb(0,0,0,0.03)] ring-1 ring-neutral-900/[0.03] dark:ring-neutral-100/[0.03]'
      } p-6 sm:p-8 space-y-6 text-neutral-900 dark:text-neutral-100`}
      onFocus={() => setIsFocused(true)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) {
          setIsFocused(false);
        }
      }}
    >
      {/* Hidden audio element */}
      <audio 
        ref={audioRef}
        src={audioUrl}
        preload="auto"
        onEnded={handleAudioEnded}
      />

      {/* Header Info */}
      <div className="flex items-center justify-between text-xs text-neutral-400 dark:text-neutral-500 font-medium">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 font-semibold uppercase tracking-wider">
            Écoute #{index + 1}
          </span>
          <span className="text-neutral-300 dark:text-neutral-600">/</span>
          <span>{total}</span>
          <span className="ml-2 px-2 py-0.5 rounded-full bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400 border border-amber-100/60 dark:border-amber-900/50 font-mono">
            {sentence.level || 'A1'}
          </span>
        </div>

        {playCount > 0 && (
          <span className="text-neutral-400 dark:text-neutral-500 flex items-center gap-1">
            <Volume2 className="w-3.5 h-3.5" />
            Écouté {playCount} {playCount === 1 ? 'fois' : 'fois'}
          </span>
        )}
      </div>

      {/* Audio Playback Controls */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-5 rounded-2xl bg-neutral-50/80 dark:bg-neutral-800/60 border border-neutral-100 dark:border-neutral-800">
        <div className="flex items-center gap-3 w-full sm:w-auto">
          {/* Main Play Button */}
          <button
            type="button"
            onClick={() => togglePlay(1.0)}
            className={`flex items-center gap-2.5 px-5 py-3 rounded-xl font-medium text-sm transition-all shadow-xs ${
              isPlaying && playbackSpeed === 1.0
                ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 shadow-neutral-900/20'
                : 'bg-white dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-700 text-neutral-800 dark:text-neutral-200 border border-neutral-200/80 dark:border-neutral-700'
            }`}
          >
            {isPlaying && playbackSpeed === 1.0 ? (
              <Pause className="w-4 h-4 fill-current" />
            ) : (
              <Play className="w-4 h-4 fill-current" />
            )}
            <span>{isPlaying && playbackSpeed === 1.0 ? 'Pause' : 'Écouter (1.0x)'}</span>
          </button>

          {/* Slow Speed Button */}
          <button
            type="button"
            onClick={() => togglePlay(0.75)}
            className={`flex items-center gap-1.5 px-4 py-3 rounded-xl font-medium text-xs transition-all ${
              isPlaying && playbackSpeed === 0.75
                ? 'bg-amber-600 text-white shadow-xs'
                : 'bg-white dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-700 text-neutral-700 dark:text-neutral-300 border border-neutral-200/80 dark:border-neutral-700'
            }`}
            title="Écouter à vitesse réduite (0.75x)"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Lent (0.75x)</span>
          </button>
        </div>

        {/* Audio Waveform Indicator */}
        <div className="flex items-center gap-1 px-3 py-1.5">
          {[40, 75, 100, 60, 30, 85, 45, 95, 50, 70].map((h, i) => (
            <div
              key={i}
              className={`w-1 rounded-full transition-all duration-300 ${
                isPlaying ? 'bg-neutral-800 dark:bg-neutral-200 animate-pulse' : 'bg-neutral-200 dark:bg-neutral-700'
              }`}
              style={{ 
                height: isPlaying ? `${Math.max(8, (h * 0.25))}px` : '6px',
                animationDelay: `${i * 75}ms` 
              }}
            />
          ))}
          <span className="text-[11px] text-neutral-400 dark:text-neutral-500 ml-2 hidden sm:inline">
            Ctrl+Espace pour rejouer
          </span>
        </div>
      </div>

      {/* Input or Result Section */}
      {!isDone ? (
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Error Feedback Banner & Marked Words (When retryEnabled is on and previous attempt had errors) */}
          {retryEnabled && attemptFeedback && (
            <div className="p-4 sm:p-5 rounded-2xl bg-amber-50/70 dark:bg-amber-950/30 border border-amber-200/80 dark:border-amber-900/50 space-y-3 animate-in fade-in duration-200">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-xs font-semibold text-amber-950 dark:text-amber-200">
                  <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
                  <span>
                    {attemptFeedback.errorCount <= 1
                      ? 'Erreur détectée dans votre transcription. Réécoutez et corrigez :'
                      : `${attemptFeedback.errorCount} erreurs détectées. Réécoutez et corrigez :`}
                  </span>
                </div>
                <span className="text-[11px] font-mono font-semibold text-amber-800 dark:text-amber-300 bg-amber-100/90 dark:bg-amber-900/60 px-2 py-0.5 rounded-full border border-amber-200/70 dark:border-amber-800/60">
                  Essai #{attemptCount}
                </span>
              </div>

              {/* Marked Words Breakdown */}
              <div className="flex flex-wrap items-center gap-2 p-3.5 rounded-xl bg-white/90 dark:bg-neutral-900/90 border border-amber-200/60 dark:border-amber-900/50 shadow-2xs">
                {attemptFeedback.diff && attemptFeedback.diff.map((item, i) => {
                  if (item.type === 'exact') {
                    return (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300 border border-emerald-200/90 dark:border-emerald-800/60 font-medium text-xs sm:text-sm flex items-center gap-1"
                        title="Mot correct"
                      >
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
                        <span>{item.word}</span>
                      </span>
                    );
                  } else if (item.type === 'accent') {
                    return (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg bg-amber-50 dark:bg-amber-950/50 text-amber-900 dark:text-amber-300 border border-amber-300 dark:border-amber-800/80 font-medium text-xs sm:text-sm flex items-center gap-1"
                        title="Accent à corriger"
                      >
                        <span>{item.word}</span>
                        <span className="text-[10px] font-bold text-amber-700 dark:text-amber-300 bg-amber-200/80 dark:bg-amber-900/80 px-1.5 py-0.5 rounded uppercase tracking-wider">
                          accent
                        </span>
                      </span>
                    );
                  } else if (item.type === 'typo') {
                    return (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg bg-orange-50 dark:bg-orange-950/50 text-orange-950 dark:text-orange-300 border border-orange-300 dark:border-orange-800/80 font-medium text-xs sm:text-sm flex items-center gap-1"
                        title="Coquille"
                      >
                        <span>{item.word}</span>
                        <span className="text-[10px] font-bold text-orange-700 dark:text-orange-300 bg-orange-200/80 dark:bg-orange-900/80 px-1.5 py-0.5 rounded uppercase tracking-wider">
                          coquille
                        </span>
                      </span>
                    );
                  } else if (item.type === 'missing') {
                    return (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg border-2 border-dashed border-rose-400 dark:border-rose-600 bg-rose-50/70 dark:bg-rose-950/40 text-rose-800 dark:text-rose-300 font-medium text-xs sm:text-sm flex items-center gap-1"
                        title="Un mot manque à cet endroit"
                      >
                        <span className="text-rose-500 font-bold">+</span>
                        <span className="italic">Mot manquant</span>
                      </span>
                    );
                  } else if (item.type === 'extra') {
                    return (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg bg-neutral-100 dark:bg-neutral-800 text-neutral-500 dark:text-neutral-400 border border-neutral-300 dark:border-neutral-700 font-medium text-xs sm:text-sm line-through flex items-center gap-1"
                        title="Mot en trop"
                      >
                        <span>{item.word}</span>
                        <span className="text-[10px] font-semibold text-neutral-400 dark:text-neutral-500 no-underline">
                          (en trop)
                        </span>
                      </span>
                    );
                  } else {
                    return (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg bg-rose-50 dark:bg-rose-950/50 text-rose-900 dark:text-rose-300 border border-rose-300 dark:border-rose-800/80 font-medium text-xs sm:text-sm flex items-center gap-1"
                        title="Mot incorrect"
                      >
                        <span>{item.word}</span>
                        <span className="text-[10px] font-bold text-rose-700 dark:text-rose-300 bg-rose-200/80 dark:bg-rose-900/80 px-1.5 py-0.5 rounded uppercase tracking-wider">
                          incorrect
                        </span>
                      </span>
                    );
                  }
                })}
              </div>
            </div>
          )}

          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isEvaluating || isSubmittingAttempt}
              placeholder="Tapez ce que vous avez entendu en français..."
              className="w-full px-5 py-4 text-base sm:text-lg bg-neutral-50/50 dark:bg-neutral-800/50 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:bg-white dark:focus:bg-neutral-800 rounded-2xl border border-neutral-200 dark:border-neutral-700 focus:border-neutral-900 dark:focus:border-neutral-300 focus:ring-4 focus:ring-neutral-900/5 dark:focus:ring-neutral-100/5 transition-all outline-hidden font-sans text-neutral-900 dark:text-neutral-100 placeholder:text-neutral-400 dark:placeholder:text-neutral-500"
            />
          </div>

          {/* Accent Helper Buttons */}
          <div className="flex items-center justify-between gap-2 pt-1">
            <div className="flex flex-wrap items-center gap-1.5">
              {FRENCH_ACCENTS.map((char) => (
                <button
                  key={char}
                  type="button"
                  onClick={() => handleAccentClick(char)}
                  disabled={isEvaluating}
                  className="w-7 h-7 sm:w-8 sm:h-8 flex items-center justify-center rounded-lg bg-neutral-100 dark:bg-neutral-800 hover:bg-neutral-200/80 dark:hover:bg-neutral-700 active:bg-neutral-300 dark:active:bg-neutral-600 text-xs sm:text-sm font-medium text-neutral-700 dark:text-neutral-300 transition-colors"
                >
                  {char}
                </button>
              ))}
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSkip}
                disabled={isEvaluating}
                className="px-3.5 py-2 rounded-xl text-xs font-medium text-neutral-500 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              >
                Passer
              </button>
              <button
                type="submit"
                disabled={isEvaluating || !inputText.trim()}
                className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl bg-neutral-900 dark:bg-neutral-100 hover:bg-neutral-800 dark:hover:bg-neutral-200 active:bg-black text-white dark:text-neutral-900 text-xs font-semibold shadow-xs transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isEvaluating ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Vérification...</span>
                  </>
                ) : (
                  <>
                    <span>Valider</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      ) : (
        /* Evaluation Results View */
        <div className="space-y-6 pt-2">
          {/* Status Banner */}
          <div className={`p-5 rounded-2xl border ${
            evalResult.is_exact
              ? 'bg-emerald-50/70 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800/60 text-emerald-950 dark:text-emerald-200'
              : evalResult.quality >= 0.75
                ? 'bg-amber-50/70 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800/60 text-amber-950 dark:text-amber-200'
                : 'bg-rose-50/70 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800/60 text-rose-950 dark:text-rose-200'
          }`}>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                {evalResult.is_exact ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                ) : evalResult.quality >= 0.75 ? (
                  <Sparkles className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400" />
                )}
                <div>
                  <h4 className="font-semibold text-sm">
                    {evalResult.is_exact 
                      ? (evalResult.attempts_taken > 1 ? `Parfait ! Réussi au ${evalResult.attempts_taken}e essai` : 'Parfait ! Transcription exacte')
                      : evalResult.quality >= 0.75 
                        ? 'Très bien ! Quelques détails à ajuster' 
                        : 'À retravailler'}
                  </h4>
                  {evalResult.first_try_score !== undefined && evalResult.first_try_score !== evalResult.composite_score && (
                    <p className="text-[11px] font-medium opacity-85 mt-0.5">
                      Score CEFR calculé sur le 1er essai : {evalResult.first_try_score}%
                    </p>
                  )}
                  {evalResult.notes?.length > 0 && (
                    <p className="text-xs opacity-80 mt-0.5">
                      {evalResult.notes.slice(0, 2).join(' · ')}
                    </p>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-3 shrink-0">
                <div className="text-right font-mono">
                  <span className="text-lg font-bold">
                    {evalResult.composite_score ?? 0}%
                  </span>
                  <span className="text-[10px] block opacity-70">Précision</span>
                </div>
                {retryEnabled && (
                  <button
                    type="button"
                    onClick={handleRetry}
                    disabled={isEvaluating}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white/90 dark:bg-neutral-800 hover:bg-white dark:hover:bg-neutral-700 text-neutral-800 dark:text-neutral-200 border border-neutral-200/80 dark:border-neutral-700 shadow-2xs transition-all hover:shadow-xs hover:scale-[1.02] active:scale-95 disabled:opacity-50"
                    title="Réessayer cette phrase"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Réessayer</span>
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Interactive Word Diff View */}
          {evalResult.diff && evalResult.diff.length > 0 && (
            <div className="space-y-2">
              <label className="text-xs font-semibold text-neutral-400 dark:text-neutral-500 uppercase tracking-wider block">
                Votre transcription
              </label>
              <div className="flex flex-wrap items-center gap-2 p-4 rounded-2xl bg-neutral-50/50 dark:bg-neutral-800/60 border border-neutral-200/70 dark:border-neutral-800">
                {evalResult.diff.map((item, i) => {
                  if (item.type === 'exact') {
                    return (
                      <span 
                        key={i} 
                        className="px-2.5 py-1 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60 font-medium text-sm"
                      >
                        {item.word}
                      </span>
                    );
                  } else if (item.type === 'accent') {
                    return (
                      <span 
                        key={i} 
                        className="group relative px-2.5 py-1 rounded-lg bg-amber-50 dark:bg-amber-950/50 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800/60 font-medium text-sm cursor-help"
                        title={`Accent attendu: ${item.ref_word}`}
                      >
                        {item.word}
                        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block px-2 py-0.5 bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 text-[10px] rounded whitespace-nowrap z-10">
                          Attendu: {item.ref_word}
                        </span>
                      </span>
                    );
                  } else if (item.type === 'typo') {
                    return (
                      <span 
                        key={i} 
                        className="group relative px-2.5 py-1 rounded-lg bg-orange-50 dark:bg-orange-950/50 text-orange-800 dark:text-orange-300 border border-orange-200 dark:border-orange-800/60 font-medium text-sm cursor-help"
                        title={`Coquille. Attendu: ${item.ref_word}`}
                      >
                        {item.word}
                        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block px-2 py-0.5 bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 text-[10px] rounded whitespace-nowrap z-10">
                          Attendu: {item.ref_word}
                        </span>
                      </span>
                    );
                  } else if (item.type === 'missing') {
                    return (
                      <span 
                        key={i} 
                        className="px-2.5 py-1 rounded-lg bg-rose-50 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300 border border-rose-200/80 dark:border-rose-800/60 font-medium text-sm line-through decoration-rose-400"
                        title="Mot manquant"
                      >
                        {item.word}
                      </span>
                    );
                  } else {
                    return (
                      <span 
                        key={i} 
                        className="px-2.5 py-1 rounded-lg bg-neutral-100 dark:bg-neutral-800 text-neutral-500 dark:text-neutral-400 border border-neutral-300 dark:border-neutral-700 text-sm line-through"
                        title="Mot en trop"
                      >
                        {item.word}
                      </span>
                    );
                  }
                })}
              </div>
            </div>
          )}

          {/* Canonical French Reference */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-neutral-400 dark:text-neutral-500 uppercase tracking-wider block">
              Phrase de référence
            </label>
            <div className="p-4 rounded-2xl bg-neutral-100/70 dark:bg-neutral-800/70 text-neutral-900 dark:text-neutral-100 text-base font-medium">
              {sentence.fr}
            </div>
          </div>

          {/* Action Footer: Translation Reveal Toggle & Retry Button */}
          <div className="space-y-3 pt-3 border-t border-neutral-100 dark:border-neutral-800">
            <div className="flex items-center justify-between gap-3">
              <button
                type="button"
                onClick={() => setShowTranslation(!showTranslation)}
                className="text-xs text-neutral-500 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 flex items-center gap-1.5 transition-colors"
              >
                <Languages className="w-3.5 h-3.5" />
                <span>{showTranslation ? 'Masquer la traduction' : 'Afficher la traduction'}</span>
              </button>

              {retryEnabled && (
                <button
                  type="button"
                  onClick={handleRetry}
                  disabled={isEvaluating}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold bg-neutral-900 dark:bg-neutral-100 hover:bg-neutral-800 dark:hover:bg-neutral-200 text-white dark:text-neutral-900 shadow-xs transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-50"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Réessayer</span>
                </button>
              )}
            </div>

            {showTranslation && (
              <p className="text-sm text-neutral-600 dark:text-neutral-300 italic bg-neutral-50 dark:bg-neutral-800/70 p-3 rounded-xl border border-neutral-100 dark:border-neutral-800">
                {sentence.es}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
