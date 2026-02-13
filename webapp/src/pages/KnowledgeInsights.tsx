import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useKnowledgeBase } from '../contexts/KnowledgeBaseContext';
import { api } from '../lib/api';
import type { KnowledgeClaimInfo, KnowledgeGraphCluster } from '../lib/api';
import { useTranslation } from 'react-i18next';
import { Lightbulb, AlertCircle, Loader2, RefreshCw } from 'lucide-react';

export default function KnowledgeInsights() {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading: authLoading, accessToken } = useAuth();
  const { selectedKB, isDefaultSelected } = useKnowledgeBase();
  const { t } = useTranslation();

  const [claims, setClaims] = useState<KnowledgeClaimInfo[]>([]);
  const [clusters, setClusters] = useState<KnowledgeGraphCluster[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const jobId = selectedKB && !selectedKB.isDefault ? Number(selectedKB.id) : null;

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login?redirect=/insights');
    }
  }, [authLoading, isAuthenticated, navigate]);

  const loadInsights = async () => {
    if (!accessToken || !jobId) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.getKnowledgeInsights(jobId, accessToken);
      setClaims(response.claims || []);
      const clusterResponse = await api.getKnowledgeGraphClusters(jobId, accessToken);
      setClusters(clusterResponse.clusters || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('insights.error_loading'));
    } finally {
      setIsLoading(false);
    }
  };

  const runInsights = async () => {
    if (!accessToken || !jobId) return;
    setIsRunning(true);
    setError(null);
    try {
      await api.runKnowledgeInsights(jobId, accessToken);
      await loadInsights();
    } catch (err) {
      setError(err instanceof Error ? err.message : t('insights.error_running'));
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    if (accessToken && jobId) {
      loadInsights();
    } else {
      setIsLoading(false);
    }
  }, [accessToken, jobId]);

  const missingCount = claims.filter(c => c.gaps.some(g => g.gap_type === 'missing_evidence')).length;
  const weakCount = claims.filter(c => c.gaps.some(g => g.gap_type === 'weak_coverage')).length;

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
            {t('insights.select_kb_title')}
          </h2>
          <p className="text-muted-foreground">
            {t('insights.select_kb_desc')}
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
              <Lightbulb className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">{t('insights.title')}</h1>
              <p className="text-sm text-muted-foreground">
                {t('insights.subtitle', { name: selectedKB?.name })}
              </p>
            </div>
          </div>
          <button
            onClick={runInsights}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                {t('insights.running')}
              </>
            ) : (
              <>
                <RefreshCw className="h-4 w-4" />
                {t('insights.run')}
              </>
            )}
          </button>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-card border border-border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">{t('insights.total_claims')}</p>
            <p className="text-2xl font-bold text-foreground">{claims.length}</p>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">{t('insights.low_support')}</p>
            <p className="text-2xl font-bold text-foreground">{missingCount}</p>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">{t('insights.sparse_support')}</p>
            <p className="text-2xl font-bold text-foreground">{weakCount}</p>
          </div>
        </div>

        {clusters.length > 0 && (
          <div className="bg-card border border-border rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-lg font-semibold text-foreground">{t('insights.clusters_title')}</h2>
                <p className="text-xs text-muted-foreground">{t('insights.clusters_subtitle')}</p>
              </div>
              <span className="text-xs text-muted-foreground">
                {clusters.length}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {clusters.slice(0, 6).map((cluster) => (
                <div key={cluster.cluster_id} className="border border-border rounded-md p-3 bg-secondary/30">
                  <div className="text-sm font-medium text-foreground">
                    {cluster.name}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {cluster.summary || t('insights.empty')}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 bg-destructive/10 border border-destructive/20 rounded-lg p-4 text-destructive flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        {isLoading ? (
          <div className="bg-card border border-border rounded-lg p-8 flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t('insights.loading')}
          </div>
        ) : (
          <div className="bg-card border border-border rounded-lg divide-y divide-border">
            {claims.length === 0 ? (
              <div className="p-8 text-muted-foreground">
                {t('insights.empty')}
              </div>
            ) : (
              claims.map((claim) => (
                <div key={claim.id} className="p-4">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-foreground">{claim.claim_text}</p>
                    <div className="flex items-center gap-2">
                      {claim.gaps.map((gap, idx) => (
                        <span
                          key={`${claim.id}-${gap.gap_type}-${idx}`}
                          className={`text-xs px-2 py-1 rounded-full border ${
                            gap.gap_type === 'missing_evidence'
                              ? 'bg-destructive/10 text-destructive border-destructive/30'
                              : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                          }`}
                        >
                          {gap.gap_type === 'missing_evidence'
                            ? t('insights.low_support')
                            : t('insights.sparse_support')}
                        </span>
                      ))}
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {t('insights.doc_id')}: {claim.doc_id}
                    {claim.paragraph_index ? ` · ${t('insights.paragraph')} ${claim.paragraph_index}` : ''}
                  </p>
                  {claim.gaps.map((gap, idx) => (
                    gap.evidence && gap.evidence.length > 0 ? (
                      <div key={`${claim.id}-evidence-${idx}`} className="mt-3 space-y-2">
                        {gap.evidence.map((ev: { doc_id?: string | null; title?: string | null; snippet?: string | null }, evIdx: number) => (
                          <div key={`${claim.id}-ev-${idx}-${evIdx}`} className="text-xs text-muted-foreground border border-border rounded-md p-2 bg-secondary/30">
                            <div className="font-medium text-foreground truncate">
                              {ev.title || ev.doc_id || t('insights.evidence_unknown')}
                            </div>
                            <div className="mt-1">
                              {ev.snippet}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : null
                  ))}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
