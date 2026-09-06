import React from 'react';
import {
  GraduationCap,
  Sparkles,
  BookOpen,
  Coffee,
  Compass,
  Rocket,
  Trophy,
  Music,
  Palette,
  Feather,
  Crown,
  Star,
  User,
  Globe
} from 'lucide-react';

export const AVATAR_PRESETS = [
  {
    id: 'owl',
    name: "L'Érudit",
    description: 'Sagesse & assiduité',
    Icon: GraduationCap,
    bgClass: 'bg-indigo-500/10 text-indigo-600 border-indigo-200/80',
    activeRing: 'ring-indigo-500',
    solidBg: 'bg-indigo-600 text-white',
  },
  {
    id: 'sparkles',
    name: "L'Inspiré",
    description: 'Créativité & intuition',
    Icon: Sparkles,
    bgClass: 'bg-amber-500/10 text-amber-600 border-amber-200/80',
    activeRing: 'ring-amber-500',
    solidBg: 'bg-amber-500 text-white',
  },
  {
    id: 'book',
    name: 'Le Lecteur',
    description: 'Littérature & vocabulaire',
    Icon: BookOpen,
    bgClass: 'bg-emerald-500/10 text-emerald-600 border-emerald-200/80',
    activeRing: 'ring-emerald-500',
    solidBg: 'bg-emerald-600 text-white',
  },
  {
    id: 'compass',
    name: "L'Explorateur",
    description: 'Curiosité & voyage',
    Icon: Compass,
    bgClass: 'bg-sky-500/10 text-sky-600 border-sky-200/80',
    activeRing: 'ring-sky-500',
    solidBg: 'bg-sky-600 text-white',
  },
  {
    id: 'coffee',
    name: 'Le Citadin',
    description: 'Café & conversation',
    Icon: Coffee,
    bgClass: 'bg-orange-500/10 text-orange-600 border-orange-200/80',
    activeRing: 'ring-orange-500',
    solidBg: 'bg-orange-600 text-white',
  },
  {
    id: 'palette',
    name: "L'Artiste",
    description: 'Sensibilité & nuance',
    Icon: Palette,
    bgClass: 'bg-rose-500/10 text-rose-600 border-rose-200/80',
    activeRing: 'ring-rose-500',
    solidBg: 'bg-rose-600 text-white',
  },
  {
    id: 'music',
    name: 'Le Mélomane',
    description: 'Écoute & prononciation',
    Icon: Music,
    bgClass: 'bg-purple-500/10 text-purple-600 border-purple-200/80',
    activeRing: 'ring-purple-500',
    solidBg: 'bg-purple-600 text-white',
  },
  {
    id: 'rocket',
    name: 'La Fusée',
    description: 'Rythme & progression',
    Icon: Rocket,
    bgClass: 'bg-violet-500/10 text-violet-600 border-violet-200/80',
    activeRing: 'ring-violet-500',
    solidBg: 'bg-violet-600 text-white',
  },
  {
    id: 'trophy',
    name: 'Le Champion',
    description: 'Défi & persévérance',
    Icon: Trophy,
    bgClass: 'bg-yellow-500/10 text-yellow-600 border-yellow-200/80',
    activeRing: 'ring-yellow-500',
    solidBg: 'bg-yellow-500 text-neutral-900',
  },
  {
    id: 'feather',
    name: 'La Plume',
    description: 'Style & précision',
    Icon: Feather,
    bgClass: 'bg-teal-500/10 text-teal-600 border-teal-200/80',
    activeRing: 'ring-teal-500',
    solidBg: 'bg-teal-600 text-white',
  },
  {
    id: 'crown',
    name: 'Le Maître',
    description: 'Excellence & maîtrise',
    Icon: Crown,
    bgClass: 'bg-amber-500/10 text-amber-700 border-amber-300',
    activeRing: 'ring-amber-600',
    solidBg: 'bg-amber-600 text-white',
  },
  {
    id: 'globe',
    name: 'Polyglotte',
    description: 'Ouverture au monde',
    Icon: Globe,
    bgClass: 'bg-blue-500/10 text-blue-600 border-blue-200/80',
    activeRing: 'ring-blue-500',
    solidBg: 'bg-blue-600 text-white',
  }
];

export const getAvatarMeta = (avatarId) => {
  const found = AVATAR_PRESETS.find(a => a.id === avatarId);
  if (found) return found;
  return {
    id: avatarId || 'default',
    name: 'Apprenant',
    description: 'Profil apprenant',
    Icon: User,
    bgClass: 'bg-neutral-100 text-neutral-700 border-neutral-200',
    activeRing: 'ring-neutral-400',
    solidBg: 'bg-neutral-800 text-white',
  };
};

const SIZES = {
  xs: {
    container: 'w-6 h-6 rounded-md border text-xs',
    icon: 'w-3.5 h-3.5',
  },
  sm: {
    container: 'w-8 h-8 rounded-lg border text-sm',
    icon: 'w-4 h-4',
  },
  md: {
    container: 'w-10 h-10 rounded-xl border text-base',
    icon: 'w-5 h-5',
  },
  lg: {
    container: 'w-14 h-14 rounded-2xl border text-xl shadow-xs',
    icon: 'w-7 h-7',
  },
  xl: {
    container: 'w-20 h-20 rounded-3xl border-2 text-2xl shadow-sm',
    icon: 'w-10 h-10',
  },
};

export default function UserAvatar({
  avatar = 'owl',
  size = 'md',
  className = '',
  ring = false,
  solid = false,
}) {
  const meta = getAvatarMeta(avatar);
  const sizeConfig = SIZES[size] || SIZES.md;
  const IconComponent = meta.Icon || User;

  return (
    <div
      className={`relative inline-flex items-center justify-center shrink-0 transition-transform ${
        sizeConfig.container
      } ${solid ? meta.solidBg : meta.bgClass} ${
        ring ? `ring-2 ring-offset-2 ${meta.activeRing}` : ''
      } ${className}`}
      title={meta.name}
    >
      <IconComponent className={sizeConfig.icon} />
    </div>
  );
}
