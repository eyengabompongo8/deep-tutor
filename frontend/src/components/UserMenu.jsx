import React, { useState, useRef, useEffect } from 'react';
import {
  User,
  ChevronDown,
  UserCheck,
  Users,
  Edit3,
  PlusCircle,
  LogOut,
  Flame,
  CheckCircle2,
  BookOpen,
  X,
  Check
} from 'lucide-react';

export default function UserMenu({
  profile,
  currentMode = 'translation',
  onSwitchUser,
  onLogout,
  onUserUpdated,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [editError, setEditError] = useState(null);

  const menuRef = useRef(null);

  const modeData = profile?.modes?.[currentMode] || profile;
  const score = modeData?.score ?? 1.20;
  const level = modeData?.level ?? 'A1';
  const streak = modeData?.streak ?? 0;
  const completed = modeData?.total_completed ?? 0;
  const userName = profile?.name || 'Default User';
  const knownWords = profile?.vocabulary?.total_known ?? 0;

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleOpenEdit = () => {
    setEditName(userName);
    setEditError(null);
    setIsEditing(true);
    setIsOpen(false);
  };

  const handleSaveProfile = async (e) => {
    e?.preventDefault();
    if (!editName.trim()) {
      setEditError('Le nom ne peut pas être vide');
      return;
    }
    setIsSaving(true);
    setEditError(null);
    try {
      const res = await fetch(`/api/users/${profile?.user_id || profile?.id || 'default_user'}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: editName.trim(),
        }),
      });
      if (!res.ok) throw new Error('Erreur lors de la mise à jour du profil');
      const updated = await res.json();
      if (onUserUpdated) {
        onUserUpdated(updated);
      }
      setIsEditing(false);
    } catch (err) {
      console.error(err);
      setEditError('Impossible de mettre à jour le profil.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="relative inline-block text-left" ref={menuRef}>
      {/* Trigger Button: User Pill */}
      <button
        type="button"
        onClick={() => setIsOpen(prev => !prev)}
        className="flex items-center gap-2 px-2.5 py-1.5 rounded-full bg-white dark:bg-neutral-800 border border-neutral-200/80 dark:border-neutral-700 hover:border-neutral-300 dark:hover:border-neutral-600 hover:bg-neutral-50/80 dark:hover:bg-neutral-750 shadow-2xs transition-all focus:outline-hidden focus:ring-2 focus:ring-neutral-400"
        aria-label="Menu utilisateur"
        title={`Profil actif: ${userName}`}
      >
        <div className="w-5 h-5 rounded-full bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 flex items-center justify-center text-[10px] font-bold uppercase">
          {userName.charAt(0) || <User className="w-3 h-3" />}
        </div>
        <span className="text-xs font-semibold text-neutral-800 dark:text-neutral-200 max-w-[110px] sm:max-w-[140px] truncate">
          {userName}
        </span>
        <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded-md bg-neutral-100 dark:bg-neutral-700 text-neutral-600 dark:text-neutral-300">
          {level}
        </span>
        <ChevronDown className={`w-3.5 h-3.5 text-neutral-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-72 sm:w-80 rounded-2xl bg-white dark:bg-neutral-900 border border-neutral-200/90 dark:border-neutral-800 shadow-xl py-2 z-50 animate-in fade-in slide-in-from-top-2 duration-150">
          {/* User Profile Header Card */}
          <div className="px-4 py-3 border-b border-neutral-100 dark:border-neutral-800 flex items-center gap-3 bg-neutral-50/50 dark:bg-neutral-850/50 rounded-t-2xl">
            <div className="w-10 h-10 rounded-full bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 flex items-center justify-center text-sm font-bold uppercase shadow-xs">
              {userName.charAt(0) || <User className="w-5 h-5" />}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="text-sm font-bold text-neutral-900 dark:text-neutral-100 truncate">
                  {userName}
                </span>
                <UserCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" title="Profil sélectionné" />
              </div>
              <div className="text-xs text-neutral-500 dark:text-neutral-400 font-mono mt-0.5">
                Niveau {level} · {score.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Quick Stats Grid */}
          <div className="grid grid-cols-3 gap-1 px-3 py-2.5 border-b border-neutral-100 dark:border-neutral-800 text-center">
            <div className="p-1.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/60">
              <div className="flex items-center justify-center gap-1 text-amber-600 dark:text-amber-400 text-xs font-bold">
                <Flame className="w-3 h-3 fill-amber-500 text-amber-500" />
                {streak}
              </div>
              <div className="text-[10px] text-neutral-400 dark:text-neutral-500 font-medium mt-0.5">Série</div>
            </div>
            <div className="p-1.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/60">
              <div className="flex items-center justify-center gap-1 text-neutral-700 dark:text-neutral-300 text-xs font-bold">
                <CheckCircle2 className="w-3 h-3 text-neutral-400" />
                {completed}
              </div>
              <div className="text-[10px] text-neutral-400 dark:text-neutral-500 font-medium mt-0.5">Complétées</div>
            </div>
            <div className="p-1.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/60">
              <div className="flex items-center justify-center gap-1 text-purple-700 dark:text-purple-400 text-xs font-bold">
                <BookOpen className="w-3 h-3 text-purple-500" />
                {knownWords}
              </div>
              <div className="text-[10px] text-neutral-400 dark:text-neutral-500 font-medium mt-0.5">Mots</div>
            </div>
          </div>

          {/* Actions */}
          <div className="p-1.5 space-y-0.5 text-xs text-neutral-700 dark:text-neutral-300 font-medium">
            <button
              onClick={() => {
                setIsOpen(false);
                if (onSwitchUser) onSwitchUser();
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors text-left"
            >
              <Users className="w-4 h-4 text-neutral-500 dark:text-neutral-400" />
              <span>Changer d'utilisateur</span>
            </button>

            <button
              onClick={handleOpenEdit}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors text-left"
            >
              <Edit3 className="w-4 h-4 text-neutral-500 dark:text-neutral-400" />
              <span>Modifier mon nom</span>
            </button>

            <button
              onClick={() => {
                setIsOpen(false);
                if (onSwitchUser) onSwitchUser(true); // create mode
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors text-left"
            >
              <PlusCircle className="w-4 h-4 text-neutral-500 dark:text-neutral-400" />
              <span>Créer un nouveau profil</span>
            </button>

            <div className="my-1 border-t border-neutral-100 dark:border-neutral-800" />

            <button
              onClick={() => {
                setIsOpen(false);
                if (onLogout) onLogout();
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl hover:bg-red-50 dark:hover:bg-red-950/40 text-red-600 dark:text-red-400 transition-colors text-left"
            >
              <LogOut className="w-4 h-4 text-red-500" />
              <span>Déconnexion</span>
            </button>
          </div>
        </div>
      )}

      {/* Edit Profile Modal */}
      {isEditing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-neutral-900/40 dark:bg-black/60 backdrop-blur-xs animate-in fade-in duration-200">
          <div className="bg-white dark:bg-neutral-900 rounded-3xl max-w-md w-full p-6 shadow-2xl border border-neutral-200 dark:border-neutral-800 animate-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between pb-4 border-b border-neutral-100 dark:border-neutral-800">
              <h3 className="text-base font-bold text-neutral-900 dark:text-neutral-100">
                Modifier mon profil
              </h3>
              <button
                type="button"
                onClick={() => setIsEditing(false)}
                className="p-1 rounded-lg text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleSaveProfile} className="mt-5 space-y-5">
              {/* Name Field */}
              <div>
                <label className="block text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase tracking-wider mb-2">
                  Nom de l'apprenant
                </label>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  maxLength={30}
                  placeholder="Ex: Sophie, Lucas..."
                  className="w-full px-3.5 py-2.5 rounded-xl border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 focus:outline-hidden focus:ring-2 focus:ring-neutral-800 dark:focus:ring-neutral-200 text-sm font-medium text-neutral-900 dark:text-neutral-100"
                  autoFocus
                />
              </div>

              {editError && (
                <div className="p-2.5 rounded-xl bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-400 text-xs font-medium">
                  {editError}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center justify-end gap-2 pt-3 border-t border-neutral-100 dark:border-neutral-800">
                <button
                  type="button"
                  onClick={() => setIsEditing(false)}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                >
                  Annuler
                </button>
                <button
                  type="submit"
                  disabled={isSaving}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 hover:bg-neutral-800 dark:hover:bg-neutral-200 transition-colors shadow-xs disabled:opacity-50"
                >
                  <Check className="w-3.5 h-3.5" />
                  <span>{isSaving ? 'Enregistrement...' : 'Sauvegarder'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
