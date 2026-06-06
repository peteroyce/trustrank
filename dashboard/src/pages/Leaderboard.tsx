import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchJSON } from '../lib/api'
import TierBadge from '../components/TierBadge'

interface LeaderEntry {
  entity_id: string; name: string; type: string; score: number; tier: string; signals: number; confidence: number
}

export default function Leaderboard() {
  const [data, setData] = useState<LeaderEntry[]>([])
  const [sort, setSort] = useState<'score' | 'signals' | 'confidence'>('score')

  useEffect(() => { fetchJSON<LeaderEntry[]>('/analytics/leaderboard?limit=50').then(setData).catch(console.error) }, [])

  const sorted = [...data].sort((a, b) => b[sort] - a[sort])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900">Leaderboard</h1>
        <div className="flex gap-2">
          {(['score', 'signals', 'confidence'] as const).map(s => (
            <button key={s} onClick={() => setSort(s)}
              className={`px-3 py-1 text-xs rounded-md ${sort === s ? 'bg-blue-100 text-blue-700 font-medium' : 'bg-gray-100 text-gray-600'}`}>
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
            <tr>
              <th className="px-4 py-3 text-left">#</th>
              <th className="px-4 py-3 text-left">Entity</th>
              <th className="px-4 py-3 text-left">Type</th>
              <th className="px-4 py-3 text-right">Score</th>
              <th className="px-4 py-3 text-center">Tier</th>
              <th className="px-4 py-3 text-right">Signals</th>
              <th className="px-4 py-3 text-right">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.map((e, i) => (
              <tr key={e.entity_id} className="hover:bg-gray-50 cursor-pointer">
                <td className="px-4 py-3 text-gray-400">{i + 1}</td>
                <td className="px-4 py-3 font-medium">
                  <Link to={`/entity/${e.entity_id}`} className="text-blue-600 hover:underline">{e.name}</Link>
                </td>
                <td className="px-4 py-3 text-gray-500">{e.type}</td>
                <td className="px-4 py-3 text-right font-mono">{e.score.toFixed(2)}</td>
                <td className="px-4 py-3 text-center"><TierBadge tier={e.tier} /></td>
                <td className="px-4 py-3 text-right text-gray-500">{e.signals}</td>
                <td className="px-4 py-3 text-right text-gray-500">{(e.confidence * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data.length === 0 && <p className="text-center text-gray-400 py-12">No data. Run POST /api/v1/admin/seed first.</p>}
      </div>
    </div>
  )
}
