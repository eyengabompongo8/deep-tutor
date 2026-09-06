import React, { useState, useEffect } from 'react';
import UserAvatar, { AVATAR_PRESETS } from './UserAvatar';
import {
  UserPlus,
  ArrowRight,
  Flame,
  CheckCircle2,
  Trash2,
  AlertCircle,
  Loader2,
  Sparkles,
  BookOpen,
  Languages,
  X
} from 'lucide-react';

const LEVEL_LABELS = {
  A1: 'Débutant',
  A2: 'Élémentaire',
  B1: 'Intermédiaire',
  B2: 'Intermédiaire Sup.',
  C1: 'Avancé',
  C2: 'Maîtrise',
};

export default function LoginScreen({
  activeUserId,
  onSelectUser,
  initialCreateMode = false,
}) {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // User creation state
  const [isCreating, setIsCreating] = useState(initialCreateMode);
  const [newName, setNewName] = useState('');
  const [newAvatar, setNewAvatar] = useState('owl');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [createError, setCreateError] = useState(null);

  // Deletion confirmation state
  const [userToDelete, setUserToDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Fetch users list
  const fetchUsers = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch('/api/users');
      if (!res.ok) throw new Error('Impossible de charger les profils');
      const data = await res.json();
      setUsers(data || []);
    } catch (err) {
      console.error(err);
      setError('Impossible de se connecter au serveur FrenchTutor.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleCreateUser = async (e) => {
    e?.preventDefault();
    const trimmed = newName.trim();
    if (!trimmed) {
      setCreateError('Veuillez entrer un nom ou prénom.');
      return;
    }
    setIsSubmitting(true);
    setCreateError(null);
    try {
      const res = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: trimmed,
          avatar: newAvatar,
        }),
      });
      if (!res.ok) throw new Error('Erreur lors de la création du profil');
      const created = await res.json();
      const createdId = created.user_id || created.id;
      if (onSelectUser) {
        onSelectUser(createdId);
      }
    } catch (err) {
      console.error(err);
      setCreateError('Échec de la création du profil.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteUser = async () => {
    if (!userToDelete) return;
    setIsDeleting(true);
    try {
      const res = await fetch(`/api/users/${userToDelete.id}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || 'Impossible de supprimer cet utilisateur');
      }
      setUserToDelete(null);
      await fetchUsers();
    } catch (err) {
      console.error(err);
      alert(err.message);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#fafafa] flex flex-col justify-between text-neutral-900">
      {/* Top Bar Header */}
      <header className="w-full py-5 px-6 border-b border-neutral-200/60 bg-white/70 backdrop-blur-md">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-neutral-900 text-white flex items-center justify-center font-bold text-sm shadow-xs">
              FR
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-bold text-neutral-900 tracking-tight">FrenchTutor</span>
              <span className="text-[10px] text-neutral-400 font-medium">Tuteur adaptatif de français</span>
            </div>
          </div>
          <div className="text-xs text-neutral-500 font-medium hidden sm:flex items-center gap-1.5">
            <Languages className="w-3.5 h-3.5" />
            <span>Español → Français</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-8 py-10 sm:py-16 flex flex-col items-center justify-center">
        {/* Welcoming Title */}
        <div className="text-center max-w-lg mb-10 sm:mb-12">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-neutral-100 text-neutral-600 text-xs font-semibold mb-3 border border-neutral-200/60">
            <Sparkles className="w-3.5 h-3.5 text-amber-500" />
            <span>Sélection de l'utilisateur</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-neutral-900 tracking-tight">
            Qui apprend aujourd'hui ?
          </h1>
          <p className="mt-2 text-sm text-neutral-500">
            Choisissez votre profil d'apprentissage pour reprendre vos exercices et conserver votre progression.
          </p>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="mb-6 p-4 rounded-2xl bg-red-50 text-red-800 text-xs flex items-center gap-3 w-full max-w-md">
            <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />
            <span className="flex-1">{error}</span>
            <button
              onClick={fetchUsers}
              className="px-2.5 py-1 rounded-lg bg-red-100 hover:bg-red-200 text-xs font-semibold"
            >
              Réessayer
            </button>
          </div>
        )}

        {/* Loading Spinner */}
        {isLoading ? (
          <div className="py-20 flex flex-col items-center justify-center gap-3 text-neutral-400">
            <Loader2 className="w-8 h-8 animate-spin text-neutral-600" />
            <span className="text-xs font-medium">Chargement des profils...</span>
          </div>
        ) : (
          <div className="w-full">
            {/* Grid of Profile Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
              {users.map((u) => {
                const isLastActive = u.id === activeUserId;
                const level = u.level || 'A1';
                const levelLabel = LEVEL_LABELS[level] || 'Niveau';
                const score = u.score || 1.20;
                const streak = u.streak || 0;
                const completed = u.total_completed || 0;

                return (
                  <div
                    key={u.id}
                    className={`group relative bg-white rounded-3xl p-6 border transition-all duration-200 flex flex-col justify-between hover:shadow-lg hover:-translate-y-0.5 ${
                      isLastActive
                        ? 'border-neutral-900 shadow-md ring-2 ring-neutral-900/10'
                        : 'border-neutral-200/80 hover:border-neutral-300'
                    }`}
                  >
                    {/* Top Badges */}
                    <div className="flex items-center justify-between mb-4">
                      {isLastActive ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-neutral-900 text-white shadow-2xs">
                          Dernière session
                        </span>
                      ) : (
                        <span className="text-[10px] font-medium text-neutral-400">
                          {levelLabel}
                        </span>
                      )}

                      {/* Delete Option (if more than 1 user) */}
                      {users.length > 1 && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setUserToDelete(u);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-neutral-300 hover:text-red-600 hover:bg-red-50 transition-all"
                          title="Supprimer ce profil"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>

                    {/* Avatar & User Name */}
                    <div className="flex flex-col items-center text-center my-2">
                      <div className="mb-3.5 transition-transform group-hover:scale-105">
                        <UserAvatar avatar={u.avatar} size="xl" ring={isLastActive} />
                      </div>
                      <h2 className="text-base sm:text-lg font-bold text-neutral-900 group-hover:text-neutral-950 truncate max-w-full">
                        {u.name}
                      </h2>
                      <div className="flex items-center gap-1.5 mt-1 text-xs font-mono font-medium text-neutral-500">
                        <span className="px-2 py-0.5 rounded-md bg-neutral-100 text-neutral-700 font-bold">
                          {level}
                        </span>
                        <span>·</span>
                        <span>{score.toFixed(2)} CEFR</span>
                      </div>
                    </div>

                    {/* Stats & Launch Button */}
                    <div className="mt-6 pt-4 border-t border-neutral-100 space-y-4">
                      <div className="grid grid-cols-2 gap-2 text-center text-xs">
                        <div className="p-2 rounded-xl bg-neutral-50">
                          <div className="flex items-center justify-center gap-1 text-amber-600 font-bold">
                            <Flame className="w-3.5 h-3.5 fill-amber-500 text-amber-500" />
                            <span>{streak}</span>
                          </div>
                          <div className="text-[10px] text-neutral-400 font-medium">Série</div>
                        </div>
                        <div className="p-2 rounded-xl bg-neutral-50">
                          <div className="flex items-center justify-center gap-1 text-neutral-700 font-bold">
                            <CheckCircle2 className="w-3.5 h-3.5 text-neutral-400" />
                            <span>{completed}</span>
                          </div>
                          <div className="text-[10px] text-neutral-400 font-medium">Phrases</div>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => onSelectUser(u.id)}
                        className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl text-xs font-bold transition-all shadow-xs ${
                          isLastActive
                            ? 'bg-neutral-900 text-white hover:bg-neutral-800'
                            : 'bg-neutral-100 text-neutral-800 hover:bg-neutral-900 hover:text-white'
                        }`}
                      >
                        <span>Continuer la pratique</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                );
              })}

              {/* Add New Profile Card */}
              <button
                type="button"
                onClick={() => {
                  setNewName('');
                  setNewAvatar('owl');
                  setCreateError(null);
                  setIsCreating(true);
                }}
                className="group border-2 border-dashed border-neutral-200 hover:border-neutral-400 rounded-3xl p-6 flex flex-col items-center justify-center text-center transition-all duration-200 bg-white/50 hover:bg-white min-h-[280px]"
              >
                <div className="w-16 h-16 rounded-2xl bg-neutral-100 group-hover:bg-neutral-900 text-neutral-400 group-hover:text-white flex items-center justify-center mb-3 transition-colors shadow-xs">
                  <UserPlus className="w-7 h-7" />
                </div>
                <span className="text-sm font-bold text-neutral-800 group-hover:text-neutral-900">
                  Nouveau profil
                </span>
                <p className="text-xs text-neutral-400 mt-1 max-w-[200px]">
                  Ajouter un nouvel apprenant avec son propre score et historique
                </p>
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Modal: Create Profile */}
      {isCreating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-neutral-900/40 backdrop-blur-xs animate-in fade-in duration-200">
          <div className="bg-white rounded-3xl max-w-lg w-full p-6 sm:p-8 shadow-2xl border border-neutral-200 animate-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between pb-4 border-b border-neutral-100">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-neutral-900 text-white flex items-center justify-center">
                  <UserPlus className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-neutral-900">
                    Créer un nouveau profil
                  </h3>
                  <p className="text-xs text-neutral-500">
                    Personnalisez votre prénom et votre icône d'apprenant
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIsCreating(false)}
                className="p-1 rounded-lg text-neutral-400 hover:text-neutral-700 hover:bg-neutral-100 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateUser} className="mt-6 space-y-6">
              {/* Name Input */}
              <div>
                <label className="block text-xs font-semibold text-neutral-700 uppercase tracking-wider mb-2">
                  Prénom ou Pseudo de l'apprenant
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Ex: Camille, Gabriel, Sofia..."
                  maxLength={30}
                  className="w-full px-4 py-3 rounded-xl border border-neutral-300 focus:outline-hidden focus:ring-2 focus:ring-neutral-800 text-sm font-medium text-neutral-900"
                  autoFocus
                />
              </div>

              {/* Avatar Picker */}
              <div>
                <label className="block text-xs font-semibold text-neutral-700 uppercase tracking-wider mb-2">
                  Choisissez votre icône d'apprentissage
                </label>
                <div className="grid grid-cols-3 sm:grid-cols-4 gap-2.5 max-h-60 overflow-y-auto p-1">
                  {AVATAR_PRESETS.map((preset) => {
                    const isSelected = newAvatar === preset.id;
                    return (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => setNewAvatar(preset.id)}
                        className={`flex flex-col items-center p-2.5 rounded-2xl border transition-all text-center ${
                          isSelected
                            ? 'border-neutral-900 bg-neutral-900/5 shadow-xs ring-2 ring-neutral-900'
                            : 'border-neutral-200 hover:border-neutral-300 hover:bg-neutral-50'
                        }`}
                      >
                        <UserAvatar avatar={preset.id} size="md" />
                        <span className="text-[11px] font-bold text-neutral-800 mt-2 truncate max-w-full">
                          {preset.name}
                        </span>
                        <span className="text-[9px] text-neutral-400 truncate max-w-full">
                          {preset.description}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {createError && (
                <div className="p-3 rounded-xl bg-red-50 text-red-700 text-xs font-medium">
                  {createError}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-100">
                <button
                  type="button"
                  onClick={() => setIsCreating(false)}
                  className="px-4 py-2.5 rounded-xl text-xs font-semibold text-neutral-600 hover:bg-neutral-100 transition-colors"
                >
                  Annuler
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-neutral-900 text-white hover:bg-neutral-800 transition-colors shadow-xs disabled:opacity-50"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Création...</span>
                    </>
                  ) : (
                    <>
                      <span>Commencer la pratique</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Delete Confirmation */}
      {userToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-neutral-900/40 backdrop-blur-xs animate-in fade-in duration-200">
          <div className="bg-white rounded-3xl max-w-sm w-full p-6 shadow-2xl border border-neutral-200 text-center animate-in zoom-in-95 duration-200">
            <div className="w-12 h-12 rounded-2xl bg-red-50 text-red-600 flex items-center justify-center mx-auto mb-3">
              <Trash2 className="w-6 h-6" />
            </div>
            <h3 className="text-base font-bold text-neutral-900">
              Supprimer le profil ?
            </h3>
            <p className="text-xs text-neutral-500 mt-2">
              Êtes-vous sûr de vouloir supprimer définitivement le profil de{' '}
              <strong className="text-neutral-800">{userToDelete.name}</strong> ? Tout son historique et vocabulaire seront effacés.
            </p>

            <div className="flex items-center justify-center gap-2.5 mt-6">
              <button
                type="button"
                onClick={() => setUserToDelete(null)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-neutral-600 hover:bg-neutral-100 transition-colors"
              >
                Annuler
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleDeleteUser}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-red-600 text-white hover:bg-red-700 transition-colors shadow-xs disabled:opacity-50"
              >
                {isDeleting ? 'Suppression...' : 'Supprimer définitivement'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="py-6 border-t border-neutral-200/60 text-center text-xs text-neutral-400">
        <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>FrenchTutor · Pratique adaptative du français</span>
          <span>CEFR Multimodal Engine</span>
        </div>
      </footer>
    </div>
  );
}
