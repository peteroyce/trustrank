import { useEffect, useRef, useState } from 'react'
import { fetchJSON } from '../lib/api'
import * as d3 from 'd3'

interface GraphData {
  nodes: string[]
  edges: { source: string; target: string; weight: number; category: string }[]
}

export default function TrustExplorer() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [entityId, setEntityId] = useState('')
  const [graph, setGraph] = useState<GraphData | null>(null)

  const loadGraph = () => {
    if (!entityId) return
    fetchJSON<GraphData>(`/entities/${entityId}/trust/graph?hops=2`).then(setGraph).catch(console.error)
  }

  useEffect(() => {
    if (!graph || !svgRef.current) return
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    const width = 800, height = 500

    const nodes = graph.nodes.map(id => ({ id }))
    const links = graph.edges.map(e => ({ source: e.source, target: e.target, weight: e.weight }))

    const sim = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links as any).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))

    const link = svg.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', '#94a3b8').attr('stroke-width', (d: any) => d.weight * 3)

    const node = svg.append('g').selectAll('circle').data(nodes).join('circle')
      .attr('r', 8).attr('fill', '#3b82f6').attr('stroke', '#fff').attr('stroke-width', 2)

    const label = svg.append('g').selectAll('text').data(nodes).join('text')
      .text((d: any) => d.id.slice(0, 8)).attr('font-size', 10).attr('fill', '#475569')
      .attr('dx', 12).attr('dy', 4)

    sim.on('tick', () => {
      link.attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y)
      node.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y)
      label.attr('x', (d: any) => d.x).attr('y', (d: any) => d.y)
    })
  }, [graph])

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Trust Graph Explorer</h1>
      <div className="flex gap-2 mb-4">
        <input value={entityId} onChange={e => setEntityId(e.target.value)} placeholder="Entity UUID"
          className="border rounded px-3 py-1.5 text-sm w-96" />
        <button onClick={loadGraph} className="bg-blue-600 text-white px-4 py-1.5 rounded text-sm font-medium">Load Graph</button>
      </div>
      <div className="bg-white rounded-lg border p-4">
        {graph ? (
          <svg ref={svgRef} width={800} height={500} />
        ) : (
          <p className="text-gray-400 text-center py-20">Enter an entity UUID and click Load Graph</p>
        )}
      </div>
    </div>
  )
}
