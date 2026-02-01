const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export interface StatsResponse {
  total_papers: number;
  total_chunks: number;
  phases: Record<string, number>;
  topics: Record<string, number>;
  year_range: {
    min: number;
    max: number;
  };
}

export interface Paper {
  doc_id: string;
  title: string;
  authors?: string;
  year?: number;
  phase?: string;
  topic?: string;
  source?: string;
}

export interface PapersResponse {
  total: number;
  papers: Paper[];
}

export interface SearchResult {
  doc_id: string;
  title: string;
  authors?: string;
  year?: number;
  phase?: string;
  topic?: string;
  chunk_text: string;
  relevance_score: number;
}

export interface QueryRequest {
  question: string;
  n_results?: number;
  synthesis_mode?: 'paragraph' | 'bullet_points' | 'structured';
  phase_filter?: string;
  topic_filter?: string;
  year_min?: number;
  year_max?: number;
}

export interface ChatResponse {
  question: string;
  answer: string;
  sources: Array<{
    citation_number: number;
    authors: string;
    year: number | string;
    title: string;
    doc_id: string;
  }>;
  model: string;
  filters_applied: Record<string, string>;
}

export interface QueryResponse {
  answer: string;
  documents: Array<{
    content: string;
    metadata: {
      doc_id: string;
      title: string;
      authors?: string;
      year?: number;
      phase?: string;
      topic_category?: string;
      filename?: string;
    };
    score: number;
  }>;
  synthesis?: string;
}

// Upload-related interfaces
export interface UploadResponse {
  success: boolean;
  doc_id?: string;
  filename: string;
  chunks_indexed: number;
  metadata?: Record<string, unknown>;
  error?: string;
}

export interface DocumentInfo {
  doc_id: string;
  title?: string;
  authors?: string;
  year?: number;
  phase?: string;
  topic_category?: string;
  filename?: string;
  total_pages?: number;
  doi?: string;
  abstract?: string;
}

export interface DocumentListResponse {
  total: number;
  documents: DocumentInfo[];
}

export interface DeleteResponse {
  success: boolean;
  doc_id: string;
  chunks_deleted: number;
  error?: string;
}

export interface UploadConfigResponse {
  enabled: boolean;
  max_file_size: number;
  allowed_extensions: string[];
  phases: Array<{
    name: string;
    full_name: string;
    description: string;
  }>;
  existing_topics: string[];
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Stats
  async getStats(): Promise<StatsResponse> {
    return this.fetch('/api/stats');
  }

  // Papers
  async getPapers(params?: {
    phase_filter?: string;
    topic_filter?: string;
    limit?: number;
  }): Promise<PapersResponse> {
    const searchParams = new URLSearchParams();
    if (params?.phase_filter) searchParams.set('phase_filter', params.phase_filter);
    if (params?.topic_filter) searchParams.set('topic_filter', params.topic_filter);
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    
    return this.fetch(`/api/papers?${searchParams.toString()}`);
  }

  // Semantic Search
  async search(params: {
    query: string;
    n_results?: number;
    phase_filter?: string;
    topic_filter?: string;
    year_min?: number;
    year_max?: number;
  }): Promise<SearchResult[]> {
    const searchParams = new URLSearchParams();
    searchParams.set('query', params.query);
    if (params.n_results) searchParams.set('n_results', params.n_results.toString());
    if (params.phase_filter) searchParams.set('phase_filter', params.phase_filter);
    if (params.topic_filter) searchParams.set('topic_filter', params.topic_filter);
    if (params.year_min) searchParams.set('year_min', params.year_min.toString());
    if (params.year_max) searchParams.set('year_max', params.year_max.toString());
    
    return this.fetch(`/api/search?${searchParams.toString()}`);
  }

  // Query (for chat) - uses LLM-powered /api/chat endpoint
  async query(request: QueryRequest): Promise<ChatResponse> {
    const searchParams = new URLSearchParams();
    searchParams.set('question', request.question);
    if (request.n_results) searchParams.set('n_sources', request.n_results.toString());
    if (request.phase_filter) searchParams.set('phase_filter', request.phase_filter);
    if (request.topic_filter) searchParams.set('topic_filter', request.topic_filter);

    return this.fetch(`/api/chat?${searchParams.toString()}`);
  }

  // Health check
  async health(): Promise<{ status: string; rag_ready: boolean }> {
    return this.fetch('/health');
  }

  // Upload configuration
  async getUploadConfig(): Promise<UploadConfigResponse> {
    return this.fetch('/api/upload/config');
  }

  // Upload PDF
  async uploadPDF(
    file: File,
    phase: string,
    topic: string,
    onProgress?: (progress: number) => void
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('phase', phase);
    formData.append('topic', topic);

    // Use XMLHttpRequest for progress tracking
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve(response);
          } catch {
            reject(new Error('Invalid response from server'));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || `HTTP ${xhr.status}`));
          } catch {
            reject(new Error(`HTTP ${xhr.status}`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error'));
      });

      xhr.open('POST', `${this.baseUrl}/api/upload`);
      xhr.send(formData);
    });
  }

  // List documents
  async listDocuments(params?: {
    phase_filter?: string;
    topic_filter?: string;
    limit?: number;
  }): Promise<DocumentListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.phase_filter) searchParams.set('phase_filter', params.phase_filter);
    if (params?.topic_filter) searchParams.set('topic_filter', params.topic_filter);
    if (params?.limit) searchParams.set('limit', params.limit.toString());

    return this.fetch(`/api/documents?${searchParams.toString()}`);
  }

  // Delete document
  async deleteDocument(docId: string): Promise<DeleteResponse> {
    const response = await fetch(`${this.baseUrl}/api/documents/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }
}

export const api = new ApiClient(API_BASE_URL);
