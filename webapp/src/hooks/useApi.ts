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
      setLoading(true);
      setError(null);
      try {
        const result = await fetchFn();
        if (mounted) {
          setData(result);
        }
      } catch (err) {
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
      mounted = false;
    };
  }, deps);

  return { data, loading, error };
}

// Hook for stats
export function useStats(accessToken?: string) {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.getStats>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function fetchStats() {
      setLoading(true);
      setError(null);
      try {
        const result = await api.getStats(accessToken);
        if (mounted) {
          setData(result);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        const isAuthError = message.includes('401') || message.toLowerCase().includes('authentication');
        if (mounted) {
          if (isAuthError) {
            setError(null);
            setData(null);
          } else {
            setError(message);
          }
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }

    fetchStats();

    return () => {
      mounted = false;
    };
  }, [accessToken]);

  return { data, loading, error };
}

// Hook for papers
export function usePapers(params?: {
  phase_filter?: string;
  topic_filter?: string;
  limit?: number;
}, accessToken?: string) {
  return useApiCall(() => api.getPapers(params, accessToken), [
    params?.phase_filter,
    params?.topic_filter,
    params?.limit,
    accessToken,
  ]);
}

// Hook for search
export function useSearch(accessToken?: string) {
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
      const data = await api.search(params, accessToken);
      setResults(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  return { results, loading, error, search };
}

// Hook for query (chat)
export function useQuery(accessToken?: string) {
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = useCallback(async (request: QueryRequest) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.query(request, accessToken);
      setResponse(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Query failed';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  return { response, loading, error, query };
}

// Hook for health check
export function useHealth() {
  return useApiCall(() => api.health(), []);
}
