import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useKnowledgeBase } from '../contexts/KnowledgeBaseContext';
import { api } from '../lib/api';
import type { KnowledgeGraphNode, KnowledgeGraphEdge } from '../lib/api';
import { useTranslation } from 'react-i18next';
import { Share2, Loader2, RefreshCw, AlertCircle } from 'lucide-react';
import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';

cytoscape.use(fcose);

function hashToColor(input: string) {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = input.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 55%)`;
}

function useCytoscape(
  nodes: KnowledgeGraphNode[],
  edges: KnowledgeGraphEdge[],
  isLoading: boolean,
  cyInstance: cytoscape.Core | null,
  setCyInstance: (cy: cytoscape.Core | null) => void
) {
  useEffect(() => {
    if (isLoading) return;
    const container = document.getElementById('graph-view');
    if (!container) return;
    if (cyInstance) {
      cyInstance.destroy();
      setCyInstance(null);
    }

    const elements = [
      ...nodes.map(node => ({
        data: {
          id: String(node.id),
          label: node.name,
          cluster: node.cluster || node.entity_type || 'concept',
        }
      })),
      ...edges.map(edge => ({
        data: {
          id: `${edge.source}-${edge.target}-${edge.relation_type}`,
          source: String(edge.source),
          target: String(edge.target),
          relation: edge.relation_type,
        }
      }))
    ];

    const cy = cytoscape({
      container,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele: any) => hashToColor(ele.data('cluster')),
            label: 'data(label)',
            color: '#e2e8f0',
            'text-outline-width': 2,
            'text-outline-color': '#0f172a',
            'font-size': 10,
            'text-max-width': 80,
            'text-wrap': 'ellipsis',
          }
        },
        {
          selector: 'edge',
          style: {
            width: 1,
            'line-color': 'rgba(148,163,184,0.35)',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'rgba(148,163,184,0.4)',
          }
        }
      ],
      layout: {
        name: 'fcose',
        animate: true,
        randomize: true,
        nodeSeparation: 80,
        idealEdgeLength: 140,
      }
    });

    setCyInstance(cy);
    return () => {
      cy.destroy();
    };
  }, [nodes, edges, isLoading, cyInstance, setCyInstance]);
}

export default function KnowledgeGraph() {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading: authLoading, accessToken } = useAuth();
  const { selectedKB, isDefaultSelected } = useKnowledgeBase();
  const { t } = useTranslation();

  const [nodes, setNodes] = useState<KnowledgeGraphNode[]>([]);
  const [edges, setEdges] = useState<KnowledgeGraphEdge[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cyInstance, setCyInstance] = useState<cytoscape.Core | null>(null);

  const jobId = selectedKB && !selectedKB.isDefault ? Number(selectedKB.id) : null;

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login?redirect=/graph');
    }
  }, [authLoading, isAuthenticated, navigate]);

  const loadGraph = async () => {
    if (!accessToken || !jobId) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.getKnowledgeGraph(jobId, accessToken);
      setNodes(response.nodes || []);
      setEdges(response.edges || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('graph.error_loading'));
    } finally {
      setIsLoading(false);
    }
  };

  const runGraph = async () => {
    if (!accessToken || !jobId) return;
    setIsRunning(true);
    setError(null);
    try {
      await api.buildKnowledgeGraph(jobId, accessToken);
      await loadGraph();
    } catch (err) {
      setError(err instanceof Error ? err.message : t('graph.error_running'));
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    if (accessToken && jobId) {
      loadGraph();
    } else {
      setIsLoading(false);
    }
  }, [accessToken, jobId]);

  const renderNodes = useMemo(() => nodes.slice(0, 200), [nodes]);
  const renderEdges = useMemo(
    () => edges.filter(e => renderNodes.find(n => n.id === e.source) && renderNodes.find(n => n.id === e.target)).slice(0, 400),
    [edges, renderNodes]
  );

  useCytoscape(renderNodes, renderEdges, isLoading, cyInstance, setCyInstance);

  if (authLoading || (!isAuthenticated && !authLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  if (isDefaultSelected || !jobId) {
    return (
      <div className="min-h-screen bg-background p-8">
        <div className="max-w-3xl mx-auto bg-card border border-border rounded-lg p-6">
          <h2 className="text-lg font-semibold text-foreground mb-2">
            {t('graph.select_kb_title')}
          </h2>
          <p className="text-muted-foreground">
            {t('graph.select_kb_desc')}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Share2 className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">{t('graph.title')}</h1>
              <p className="text-sm text-muted-foreground">
                {t('graph.subtitle', { name: selectedKB?.name })}
              </p>
            </div>
          </div>
          <button
            onClick={runGraph}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                {t('graph.running')}
              </>
            ) : (
              <>
                <RefreshCw className="h-4 w-4" />
                {t('graph.run')}
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mb-4 bg-destructive/10 border border-destructive/20 rounded-lg p-4 text-destructive flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        {isLoading ? (
          <div className="bg-card border border-border rounded-lg p-8 flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t('graph.loading')}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-3">{t('graph.visual')}</h3>
              <div
                id="graph-view"
                className="bg-background rounded-lg border border-border"
                style={{ height: 420 }}
              />
              <p className="text-xs text-muted-foreground mt-2">
                {t('graph.visual_note')}
              </p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-3">{t('graph.node_list')}</h3>
              <div className="max-h-[360px] overflow-y-auto space-y-2">
                {nodes.length === 0 ? (
                  <p className="text-sm text-muted-foreground">{t('graph.empty')}</p>
                ) : (
                  nodes.map((node) => (
                    <div key={node.id} className="text-sm text-foreground border border-border rounded-md p-2">
                      <div className="font-medium">{node.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {node.entity_type}{node.cluster ? ` · ${node.cluster}` : ''}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
