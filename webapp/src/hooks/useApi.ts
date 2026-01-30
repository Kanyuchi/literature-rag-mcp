import { useState, useEffect, useCallback } from 'react';
import { api, type SearchResult, type ChatResponse, type QueryRequest } from '@/lib/api';

// Generic hook for API calls
function useApiCall<T>(fetchFn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function fetch() {
      console.log('[useApiCall] Starting fetch...');
      setLoading(true);
      setError(null);
      try {
        const result = await fetchFn();
        console.log('[useApiCall] Got result:', result);
        if (mounted) {
          console.log('[useApiCall] Setting data...');
          setData(result);
        } else {
          console.log('[useApiCall] Component unmounted, skipping setData');
        }
      } catch (err) {
        console.error('[useApiCall] Error:', err);
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }

    fetch();

    return () => {
      console.log('[useApiCall] Cleanup - setting mounted=false');
      mounted = false;
    };
  }, deps);

  return { data, loading, error };
}

// Hook for stats
export function useStats() {
  return useApiCall(() => api.getStats(), []);
}

// Hook for papers
export function usePapers(params?: {
  phase_filter?: string;
  topic_filter?: string;
  limit?: number;
}) {
  return useApiCall(() => api.getPapers(params), [
    params?.phase_filter,
    params?.topic_filter,
    params?.limit,
  ]);
}

// Hook for search
export function useSearch() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(async (params: {
    query: string;
    n_results?: number;
    phase_filter?: string;
    topic_filter?: string;
    year_min?: number;
    year_max?: number;
  }) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.search(params);
      setResults(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { results, loading, error, search };
}

// Hook for query (chat)
export function useQuery() {
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = useCallback(async (request: QueryRequest) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.query(request);
      setResponse(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Query failed';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { response, loading, error, query };
}

// Hook for health check
export function useHealth() {
  return useApiCall(() => api.health(), []);
}
