import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Leaderboard from './pages/Leaderboard'
import EntityDetail from './pages/EntityDetail'
import TrustExplorer from './pages/TrustExplorer'
import Detection from './pages/Detection'
import Analytics from './pages/Analytics'

const NAV = [
  { path: '/', label: 'Leaderboard' },
  { path: '/trust', label: 'Trust Graph' },
  { path: '/detection', label: 'Detection' },
  { path: '/analytics', label: 'Analytics' },
]

export default function App() {
  const loc = useLocation()
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center gap-8">
          <span className="text-lg font-bold text-blue-600">trustrank</span>
          <div className="flex gap-4">
            {NAV.map(n => (
              <Link key={n.path} to={n.path}
                className={`text-sm font-medium px-3 py-1.5 rounded-md transition ${
                  loc.pathname === n.path ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:text-gray-900'
                }`}>{n.label}</Link>
            ))}
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-6 py-6">
        <Routes>
          <Route path="/" element={<Leaderboard />} />
          <Route path="/entity/:id" element={<EntityDetail />} />
          <Route path="/trust" element={<TrustExplorer />} />
          <Route path="/detection" element={<Detection />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </main>
    </div>
  )
}
