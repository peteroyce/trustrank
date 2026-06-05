const COLORS: Record<string, string> = {
  platinum: 'bg-violet-100 text-violet-800',
  gold: 'bg-amber-100 text-amber-800',
  silver: 'bg-gray-200 text-gray-700',
  bronze: 'bg-orange-100 text-orange-700',
  untrusted: 'bg-red-100 text-red-700',
}

export default function TierBadge({ tier }: { tier: string }) {
  return <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${COLORS[tier] || 'bg-gray-100'}`}>{tier}</span>
}
