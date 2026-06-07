import { useEffect, useState } from 'react'
import { fetchJSON } from '../lib/api'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'

interface Overview { total_entities: number; total_signals: number; tier_distribution: Record<string, number>; signals_today: number; dampened_count: number }

const COLORS: Record<string, string> = { platinum: '#8b5cf6', gold: '#f59e0b', silver: '#9ca3af', bronze: '#ea580c', untrusted: '#ef4444' }

export default function Analytics() {
  const [data, setData] = useState<Overview | null>(null)
  useEffect(() => { fetchJSON<Overview>('/analytics/overview').then(setData).catch(console.error) }, [])

  if (!data) return <p className="text-gray-400">Loading...</p>

  const pieData = Object.entries(data.tier_distribution).map(([name, value]) => ({ name, value }))

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Analytics</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border p-4">
          <h2 className="font-semibold mb-2">Tier Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                {pieData.map(entry => <Cell key={entry.name} fill={COLORS[entry.name] || '#94a3b8'} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <h2 className="font-semibold mb-2">System Overview</h2>
          <div className="space-y-3 text-sm">
            <Stat label="Total Entities" value={data.total_entities} />
            <Stat label="Total Signals" value={data.total_signals} />
            <Stat label="Signals Today" value={data.signals_today} />
            <Stat label="Dampened Signals" value={data.dampened_count} />
            <Stat label="Dampened Rate" value={`${data.total_signals > 0 ? ((data.dampened_count / data.total_signals) * 100).toFixed(1) : 0}%`} />
          </div>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between border-b border-gray-100 pb-2">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono font-medium">{typeof value === 'number' ? value.toLocaleString() : value}</span>
    </div>
  )
}
