import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Search as SearchIcon, Loader2, FileText, ChevronDown, Filter, Database, Folder } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useStats } from '@/hooks/useApi';
import { useKnowledgeBase } from '@/contexts/KnowledgeBaseContext';
import { useAuth } from '@/contexts/AuthContext';
import { api } from '@/lib/api';
import type { SearchResult, JobQueryResult } from '@/lib/api';
import { useTranslation } from 'react-i18next';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
    },
  },
};

// Unified result type for display
interface DisplayResult {
  doc_id: string;
  title: string;
  authors?: string;
  year?: number;
  phase?: string;
  topic?: string;
  chunk_text: string;
  relevance_score: number;
}

export default function Search() {
  const [query, setQuery] = useState('');
  const [phaseFilter, setPhaseFilter] = useState<string>('');
  const [topicFilter, setTopicFilter] = useState<string>('');
  const [results, setResults] = useState<DisplayResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const { selectedKB, isDefaultSelected } = useKnowledgeBase();
  const { accessToken } = useAuth();
  const { data: defaultStats } = useStats(accessToken || undefined);
  const { t } = useTranslation();

  // Job stats for non-default KB
  const [jobStats, setJobStats] = useState<{ phases: Record<string, number>; topics: Record<string, number> } | null>(null);

  // Load job stats when KB changes
  useEffect(() => {
    if (!isDefaultSelected && selectedKB && accessToken) {
      api.getJobStats(selectedKB.id as number, accessToken)
        .then(stats => setJobStats({ phases: stats.phases, topics: stats.topics }))
        .catch(err => console.error('Failed to load job stats:', err));
    } else {
      setJobStats(null);
    }
  }, [selectedKB, isDefaultSelected, accessToken]);

  // Use appropriate stats based on selected KB
  const stats = isDefaultSelected ? defaultStats : jobStats;
  const phases = stats ? Object.keys(stats.phases) : [];
  const topics = stats ? Object.keys(stats.topics) : [];

  // Reset filters and results when KB changes
  useEffect(() => {
    setPhaseFilter('');
    setTopicFilter('');
    setResults([]);
    setHasSearched(false);
    setError(null);
  }, [selectedKB?.id]);

  const handleSearch = useCallback(async () => {
    if (!query.trim() || !selectedKB) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      if (isDefaultSelected) {
        // Search default collection
        const searchResults: SearchResult[] = await api.search({
          query: query.trim(),
          n_results: 10,
          phase_filter: phaseFilter || undefined,
          topic_filter: topicFilter || undefined,
        }, accessToken || undefined);

        setResults(searchResults.map(r => ({
          doc_id: r.doc_id,
          title: r.title,
          authors: r.authors,
          year: r.year,
          phase: r.phase,
          topic: r.topic,
          chunk_text: r.chunk_text,
          relevance_score: r.relevance_score,
        })));
      } else {
        // Search job collection using query endpoint
        const response = await api.queryJob(
          selectedKB.id as number,
          query.trim(),
          {
            n_sources: 10,
            phase_filter: phaseFilter || undefined,
            topic_filter: topicFilter || undefined,
          },
          accessToken || undefined
        );

        if (response.results && response.results.length > 0) {
          setResults(response.results.map((r: JobQueryResult) => ({
            doc_id: r.metadata.doc_id,
            title: r.metadata.title || 'Untitled',
            authors: r.metadata.authors,
            year: r.metadata.year,
            phase: r.metadata.phase,
            topic: r.metadata.topic_category,
            chunk_text: r.content,
            relevance_score: r.score,
          })));
        } else {
          setResults([]);
        }
      }
    } catch (err) {
      console.error('Search error:', err);
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, selectedKB, isDefaultSelected, accessToken, phaseFilter, topicFilter]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-[calc(100vh-72px)] bg-background px-4 md:px-8 lg:px-12 py-6"
    >
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div
          variants={itemVariants}
          className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <SearchIcon className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-white">{t('search.title')}</h1>
              {/* Show which KB is being searched */}
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                {isDefaultSelected ? (
                  <Database className="w-3 h-3" />
                ) : (
                  <Folder className="w-3 h-3" />
                )}
                <span>{t('search.searching', { name: selectedKB?.name })}</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Search Input */}
        <motion.div variants={itemVariants} className="mb-8">
          <Card className="p-6 bg-card border-border">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1 relative">
                <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  placeholder={t('search.search_placeholder', { name: isDefaultSelected ? t('chat.the_literature') : selectedKB?.name })}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="pl-12 h-12 bg-secondary/50 border-border focus:border-primary text-lg"
                />
              </div>

              {/* Phase Filter */}
              {phases.length > 0 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2 h-12">
                      <Filter className="w-4 h-4" />
                      {phaseFilter || t('search.phase')}
                      <ChevronDown className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="bg-card border-border">
                    <DropdownMenuItem onClick={() => setPhaseFilter('')}>{t('search.all_phases')}</DropdownMenuItem>
                    {phases.map(phase => (
                      <DropdownMenuItem key={phase} onClick={() => setPhaseFilter(phase)}>
                        {phase}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              {/* Topic Filter */}
              {topics.length > 0 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2 h-12">
                      <Filter className="w-4 h-4" />
                      {topicFilter || t('search.topic')}
                      <ChevronDown className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="bg-card border-border">
                    <DropdownMenuItem onClick={() => setTopicFilter('')}>{t('search.all_topics')}</DropdownMenuItem>
                    {topics.map(topic => (
                      <DropdownMenuItem key={topic} onClick={() => setTopicFilter(topic)}>
                        {topic}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              <Button
                onClick={handleSearch}
                disabled={loading || !query.trim()}
                className="h-12 px-8 bg-white text-background hover:bg-white/90"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  t('search.button')
                )}
              </Button>
            </div>
          </Card>
        </motion.div>

        {/* Error State */}
        {error && (
          <motion.div variants={itemVariants} className="mb-8">
            <Card className="p-6 bg-destructive/10 border-destructive/50">
              <p className="text-destructive">{t('search.search_failed', { error })}</p>
            </Card>
          </motion.div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <motion.div variants={itemVariants}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">
                {t('search.results', { count: results.length })}
              </h2>
            </div>

            <div className="space-y-4">
              {results.map((result, index) => (
                <Card key={index} className="p-6 bg-card border-border hover:border-primary/50 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <FileText className="w-5 h-5 text-primary" />
                      <h3 className="text-lg font-semibold text-white">{result.title}</h3>
                    </div>
                    <Badge variant="secondary" className="bg-primary/20 text-primary">
                      {t('search.score', { score: (result.relevance_score * 100).toFixed(1) })}
                    </Badge>
                  </div>

                  <div className="flex flex-wrap gap-2 mb-3">
                    {result.authors && (
                      <Badge variant="secondary" className="bg-secondary/50">
                        {result.authors}
                      </Badge>
                    )}
                    {result.year && (
                      <Badge variant="secondary" className="bg-secondary/50">
                        {result.year}
                      </Badge>
                    )}
                    {result.phase && (
                      <Badge variant="secondary" className="bg-secondary/50">
                        {result.phase}
                      </Badge>
                    )}
                    {result.topic && (
                      <Badge variant="secondary" className="bg-secondary/50">
                        {result.topic}
                      </Badge>
                    )}
                  </div>

                  <p className="text-muted-foreground text-sm line-clamp-3">
                    {result.chunk_text}
                  </p>
                </Card>
              ))}
            </div>
          </motion.div>
        )}

        {/* Empty State after search */}
        {!loading && results.length === 0 && hasSearched && !error && (
          <motion.div
            variants={itemVariants}
            className="flex flex-col items-center justify-center py-32"
          >
            <div className="w-16 h-16 rounded-full bg-secondary/50 flex items-center justify-center mb-4">
              <SearchIcon className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground text-sm">{t('search.no_results')}</p>
          </motion.div>
        )}

        {/* Initial State */}
        {!loading && results.length === 0 && !hasSearched && (
          <motion.div
            variants={itemVariants}
            className="flex flex-col items-center justify-center py-32"
          >
            <div className="w-16 h-16 rounded-full bg-secondary/50 flex items-center justify-center mb-4">
              <SearchIcon className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground text-sm">
              {t('search.empty_prompt', { name: isDefaultSelected ? t('chat.the_literature') : `"${selectedKB?.name}"` })}
            </p>
            {selectedKB && (
              <p className="text-xs mt-2 text-muted-foreground/70">
                {selectedKB.document_count} {t('kb.docs')} · {selectedKB.chunk_count.toLocaleString()} {t('kb.chunks')}
              </p>
            )}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
