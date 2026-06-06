import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchJSON } from '../lib/api'
import TierBadge from '../components/TierBadge'
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'

interface ScoreData {
  entity_id: string; overall: number; confidence: number; tier: string
  dimensions: Record<string, { score: number; wilson_lower: number; trend: string; signals: number; confidence: number }>
  breakdown: Record<string, number>; alerts: { type: string; detail: string; severity: string }[]
  counterfactual: Record<string, number>
}

export default function EntityDetail() {
  const { id } = useParams()
  const [score, setScore] = useState<ScoreData | null>(null)

  useEffect(() => {
    if (id) fetchJSON<ScoreData>(`/entities/${id}/score`).then(setScore).catch(console.error)
  }, [id])

  if (!score) return <p className="text-gray-400">Loading...</p>

  const radarData = Object.entries(score.dimensions).map(([k, v]) => ({ dim: k, score: v.score, fullMark: 5 }))

  return (
    <div>
      <Link to="/" className="text-sm text-blue-600 hover:underline mb-4 block">&larr; Back to Leaderboard</Link>
      <div className="flex items-center gap-4 mb-6">
        <h1 className="text-2xl font-bold">{id?.slice(0, 8)}...</h1>
        <TierBadge tier={score.tier} />
        <span className="text-3xl font-mono font-bold text-blue-600">{score.overall.toFixed(2)}</span>
        <span className="text-sm text-gray-400">confidence: {(score.confidence * 100).toFixed(0)}%</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Radar */}
        <div className="bg-white rounded-lg border p-4">
          <h2 className="font-semibold mb-2">Dimension Scores</h2>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={radarData}>
              <PolarGrid /><PolarAngleAxis dataKey="dim" />
              <Radar dataKey="score" stroke="#2563eb" fill="#3b82f6" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Breakdown */}
        <div className="bg-white rounded-lg border p-4">
          <h2 className="font-semibold mb-2">Score Breakdown</h2>
          <div className="space-y-2 text-sm">
            {Object.entries(score.breakdown).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-gray-600">{k}</span>
                <span className={`font-mono ${typeof v === 'number' && v < 0 ? 'text-red-600' : 'text-gray-900'}`}>
                  {typeof v === 'number' ? (v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3)) : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Dimensions table */}
        <div className="bg-white rounded-lg border p-4 md:col-span-2">
          <h2 className="font-semibold mb-2">Dimensions</h2>
          <table className="w-full text-sm">
            <thead><tr className="text-gray-500 text-xs uppercase">
              <th className="text-left py-2">Dimension</th><th className="text-right">Score</th>
              <th className="text-right">Wilson</th><th className="text-center">Trend</th>
              <th className="text-right">Signals</th><th className="text-right">Confidence</th>
            </tr></thead>
            <tbody>
              {Object.entries(score.dimensions).map(([k, v]) => (
                <tr key={k} className="border-t">
                  <td className="py-2 font-medium">{k}</td>
                  <td className="text-right font-mono">{v.score.toFixed(2)}</td>
                  <td className="text-right font-mono text-gray-500">{v.wilson_lower.toFixed(3)}</td>
                  <td className="text-center">{v.trend === 'improving' ? '📈' : v.trend === 'declining' ? '📉' : '➡️'}</td>
                  <td className="text-right text-gray-500">{v.signals}</td>
                  <td className="text-right text-gray-500">{(v.confidence * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Alerts */}
        {score.alerts.length > 0 && (
          <div className="bg-white rounded-lg border p-4 md:col-span-2">
            <h2 className="font-semibold mb-2 text-red-600">Active Alerts</h2>
            {score.alerts.map((a, i) => (
              <div key={i} className="bg-red-50 border border-red-200 rounded p-3 mb-2">
                <span className="font-medium text-red-800">{a.type}</span>
                <span className="text-red-600 ml-2">{a.detail}</span>
              </div>
            ))}
          </div>
        )}

        {/* Counterfactual */}
        <div className="bg-white rounded-lg border p-4 md:col-span-2">
          <h2 className="font-semibold mb-2">What-If Analysis</h2>
          <div className="grid grid-cols-3 gap-4 text-sm">
            {Object.entries(score.counterfactual).map(([k, v]) => (
              <div key={k} className="bg-gray-50 rounded p-3">
                <div className="text-gray-500 text-xs">{k.replace(/_/g, ' ')}</div>
                <div className="font-mono text-lg">{typeof v === 'number' ? v.toFixed(2) : String(v)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
