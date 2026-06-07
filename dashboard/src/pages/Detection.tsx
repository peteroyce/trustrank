import { useEffect, useState } from 'react'
import { fetchJSON } from '../lib/api'

interface Overview { total_entities: number; total_signals: number; dampened_count: number; signals_today: number; tier_distribution: Record<string, number> }

export default function Detection() {
  const [data, setData] = useState<Overview | null>(null)
  useEffect(() => { fetchJSON<Overview>('/analytics/overview').then(setData).catch(console.error) }, [])

  if (!data) return <p className="text-gray-400">Loading...</p>

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Detection Dashboard</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Card label="Total Entities" value={data.total_entities} />
        <Card label="Total Signals" value={data.total_signals} />
        <Card label="Signals Today" value={data.signals_today} />
        <Card label="Dampened Signals" value={data.dampened_count} color="red" />
      </div>
      <div className="bg-white rounded-lg border p-4">
        <h2 className="font-semibold mb-3">Detection Stats</h2>
        <div className="text-sm text-gray-600">
          <p>Dampened rate: {data.total_signals > 0 ? ((data.dampened_count / data.total_signals) * 100).toFixed(1) : 0}%</p>
          <p className="mt-2">Detection systems active: CUSUM Burst, Coordination (TF-IDF + Poisson), Source Credibility, Reciprocal Network</p>
        </div>
      </div>
    </div>
  )
}

function Card({ label, value, color = 'blue' }: { label: string; value: number; color?: string }) {
  return (
    <div className="bg-white rounded-lg border p-4">
      <div className="text-xs text-gray-500 uppercase">{label}</div>
      <div className={`text-2xl font-bold ${color === 'red' ? 'text-red-600' : 'text-gray-900'}`}>{value.toLocaleString()}</div>
    </div>
  )
}
