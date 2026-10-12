import React, { useState, useEffect } from 'react';
import {
  User,
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
      setError('Impossible de se connecter au serveur DeepTutor.');
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
    <div className="min-h-screen bg-neutral-900 flex flex-col justify-between p-4 sm:p-8 selection:bg-neutral-800 selection:text-white">
      {/* Background Decor */}
      <div className="fixed inset-0 pointer-events-none opacity-25 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-neutral-800 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-neutral-800 rounded-full blur-3xl" />
      </div>

      {/* Top Header */}
      <header className="relative z-10 max-w-5xl w-full mx-auto flex items-center justify-between py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-white text-neutral-900 flex items-center justify-center shadow-lg font-black text-lg">
            DT
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight leading-tight">
              DeepTutor
            </h1>
            <p className="text-xs text-neutral-400">
              Apprentissage adaptatif du français
            </p>
          </div>
        </div>

        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-neutral-800/80 border border-neutral-700/60 text-xs text-neutral-300 font-mono">
          <Languages className="w-3.5 h-3.5 text-neutral-400" />
          <span>Multimodal · Adaptatif</span>
        </div>
      </header>

      {/* Main Container */}
      <main className="relative z-10 max-w-5xl w-full mx-auto my-auto py-8">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-20 text-neutral-400 gap-3">
            <Loader2 className="w-8 h-8 animate-spin text-neutral-500" />
            <span className="text-sm font-medium">Chargement des profils d'apprenants...</span>
          </div>
        ) : error ? (
          <div className="max-w-md mx-auto p-6 rounded-3xl bg-red-950/40 border border-red-800/60 text-center space-y-4">
            <AlertCircle className="w-8 h-8 text-red-400 mx-auto" />
            <p className="text-sm text-red-200">{error}</p>
            <button
              onClick={fetchUsers}
              className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white text-xs font-semibold transition-colors"
            >
              Réessayer
            </button>
          </div>
        ) : (
          <div className="space-y-8">
            <div className="text-center space-y-2">
              <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                Qui apprend le français aujourd'hui ?
              </h2>
              <p className="text-sm text-neutral-400 max-w-md mx-auto">
                Choisissez votre profil pour reprendre vos exercices personnalisés et votre progression.
              </p>
            </div>

            {/* Profile Selection Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6">
              {users.map((u) => {
                const isLastActive = activeUserId === u.id;
                const modeData = u?.modes?.translation || u;
                const level = modeData?.level ?? 'A1';
                const streak = modeData?.streak ?? 0;
                const completed = modeData?.total_completed ?? 0;

                return (
                  <div
                    key={u.id}
                    className={`group relative rounded-3xl p-6 bg-white dark:bg-neutral-900 transition-all duration-200 hover:-translate-y-1 hover:shadow-xl flex flex-col justify-between ${
                      isLastActive
                        ? 'ring-2 ring-white/80 dark:ring-neutral-400 shadow-2xl bg-white dark:bg-neutral-900'
                        : 'border border-neutral-200/90 dark:border-neutral-800 hover:border-neutral-300 dark:hover:border-neutral-700'
                    }`}
                  >
                    {/* Top Card Badge / Delete Button */}
                    <div className="flex items-center justify-between mb-2">
                      {isLastActive ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 text-[10px] font-bold border border-emerald-200/50 dark:border-emerald-800/50">
                          <Sparkles className="w-3 h-3" />
                          Actif récemment
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono text-neutral-400 dark:text-neutral-500 uppercase tracking-wider">
                          Apprenant
                        </span>
                      )}

                      {users.length > 1 && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setUserToDelete(u);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-neutral-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/50 transition-all"
                          title="Supprimer ce profil"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>

                    {/* User Avatar & Name */}
                    <div className="flex flex-col items-center text-center my-2">
                      <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-xl font-bold uppercase transition-transform group-hover:scale-105 shadow-xs mb-3.5 ${
                        isLastActive ? 'bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 ring-4 ring-neutral-900/10 dark:ring-neutral-100/10' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200'
                      }`}>
                        {u.name ? u.name.charAt(0) : <User className="w-8 h-8" />}
                      </div>
                      <h2 className="text-base sm:text-lg font-bold text-neutral-900 dark:text-neutral-100 group-hover:text-neutral-950 dark:group-hover:text-white truncate max-w-full">
                        {u.name}
                      </h2>
                      <div className="flex items-center gap-1.5 mt-1 text-xs font-mono font-medium text-neutral-500 dark:text-neutral-400">
                        <span className="px-2 py-0.5 rounded-md bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 font-bold">
                          {level}
                        </span>
                        <span>·</span>
                        <span>{LEVEL_LABELS[level] || level}</span>
                      </div>
                    </div>

                    {/* Stats & Launch Button */}
                    <div className="mt-6 pt-4 border-t border-neutral-100 dark:border-neutral-800 space-y-4">
                      <div className="grid grid-cols-2 gap-2 text-center text-xs">
                        <div className="p-2 rounded-xl bg-neutral-50 dark:bg-neutral-800/60 border border-neutral-100 dark:border-neutral-800">
                          <div className="flex items-center justify-center gap-1 text-amber-600 dark:text-amber-400 font-bold">
                            <Flame className="w-3.5 h-3.5 fill-amber-500 text-amber-500" />
                            <span>{streak}</span>
                          </div>
                          <div className="text-[10px] text-neutral-400 dark:text-neutral-500 font-medium">Série</div>
                        </div>
                        <div className="p-2 rounded-xl bg-neutral-50 dark:bg-neutral-800/60 border border-neutral-100 dark:border-neutral-800">
                          <div className="flex items-center justify-center gap-1 text-neutral-700 dark:text-neutral-300 font-bold">
                            <CheckCircle2 className="w-3.5 h-3.5 text-neutral-400 dark:text-neutral-500" />
                            <span>{completed}</span>
                          </div>
                          <div className="text-[10px] text-neutral-400 dark:text-neutral-500 font-medium">Phrases</div>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => onSelectUser(u.id)}
                        className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl text-xs font-bold transition-all shadow-xs ${
                          isLastActive
                            ? 'bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 hover:bg-neutral-800 dark:hover:bg-neutral-200'
                            : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200 hover:bg-neutral-900 dark:hover:bg-neutral-100 hover:text-white dark:hover:text-neutral-900'
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
                  setCreateError(null);
                  setIsCreating(true);
                }}
                className="group border-2 border-dashed border-neutral-700 hover:border-neutral-500 rounded-3xl p-6 flex flex-col items-center justify-center text-center transition-all duration-200 bg-neutral-800/40 hover:bg-neutral-800/70 min-h-[280px]"
              >
                <div className="w-16 h-16 rounded-2xl bg-neutral-800 group-hover:bg-neutral-100 text-neutral-400 group-hover:text-neutral-900 flex items-center justify-center mb-3 transition-colors shadow-xs">
                  <UserPlus className="w-7 h-7" />
                </div>
                <span className="text-sm font-bold text-neutral-200 group-hover:text-white">
                  Nouveau profil
                </span>
                <p className="text-xs text-neutral-400 mt-1 max-w-[200px]">
                  Ajouter un nouvel apprenant avec son propre niveau et historique
                </p>
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Modal: Create Profile */}
      {isCreating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-in fade-in duration-200">
          <div className="bg-white dark:bg-neutral-900 rounded-3xl max-w-lg w-full p-6 sm:p-8 shadow-2xl border border-neutral-200 dark:border-neutral-800 animate-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between pb-4 border-b border-neutral-100 dark:border-neutral-800">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 flex items-center justify-center">
                  <UserPlus className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-neutral-900 dark:text-neutral-100">
                    Créer un nouveau profil
                  </h3>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Entrez votre prénom ou pseudo pour commencer
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIsCreating(false)}
                className="p-1 rounded-lg text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateUser} className="mt-6 space-y-6">
              {/* Name Input */}
              <div>
                <label className="block text-xs font-semibold text-neutral-700 dark:text-neutral-300 uppercase tracking-wider mb-2">
                  Prénom ou Pseudo de l'apprenant
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Ex: Camille, Gabriel, Sofia..."
                  maxLength={30}
                  className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 focus:outline-hidden focus:ring-2 focus:ring-neutral-800 dark:focus:ring-neutral-200 text-sm font-medium"
                  autoFocus
                />
              </div>

              {createError && (
                <div className="p-3 rounded-xl bg-red-50 dark:bg-red-950/60 border border-red-200 dark:border-red-900/50 text-red-700 dark:text-red-300 text-xs font-medium">
                  {createError}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-100 dark:border-neutral-800">
                <button
                  type="button"
                  onClick={() => setIsCreating(false)}
                  className="px-4 py-2.5 rounded-xl text-xs font-semibold text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                >
                  Annuler
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 hover:bg-neutral-800 dark:hover:bg-neutral-200 transition-colors shadow-xs disabled:opacity-50"
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
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-in fade-in duration-200">
          <div className="bg-white dark:bg-neutral-900 rounded-3xl max-w-sm w-full p-6 shadow-2xl border border-neutral-200 dark:border-neutral-800 text-center animate-in zoom-in-95 duration-200">
            <div className="w-12 h-12 rounded-2xl bg-red-50 dark:bg-red-950/60 text-red-600 dark:text-red-400 flex items-center justify-center mx-auto mb-3">
              <Trash2 className="w-6 h-6" />
            </div>
            <h3 className="text-base font-bold text-neutral-900 dark:text-neutral-100">
              Supprimer le profil ?
            </h3>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-2">
              Êtes-vous sûr de vouloir supprimer définitivement le profil de{' '}
              <strong className="text-neutral-800 dark:text-neutral-200">{userToDelete.name}</strong> ? Tout son historique et vocabulaire seront effacés.
            </p>

            <div className="flex items-center justify-center gap-2.5 mt-6">
              <button
                type="button"
                onClick={() => setUserToDelete(null)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
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
          <span>DeepTutor · Pratique adaptative du français</span>
          <span>CEFR Multimodal Engine</span>
        </div>
      </footer>
    </div>
  );
}
