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
  deep_analysis?: boolean;
}

export interface PipelineStats {
  llm_calls: number;
  retrieval_attempts: number;
  validation_passed: boolean | null;
  total_time_ms: number;
  evaluation_scores: {
    relevance: number;
    coverage: number;
    diversity: number;
    overall: number;
  } | null;
  retries: {
    retrieval: number;
    generation: number;
  };
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
  complexity: 'simple' | 'medium' | 'complex';
  pipeline_stats: PipelineStats;
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

// Async upload interfaces
export interface AsyncUploadResponse {
  task_id: string;
  status: string;
  message: string;
  filename: string;
  phase: string;
  topic: string;
}

export interface TaskStatusResponse {
  task_id: string;
  filename: string;
  phase: string;
  topic: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  result: {
    doc_id: string;
    filename: string;
    chunks_indexed: number;
    metadata: Record<string, unknown>;
  } | null;
  error: string | null;
}

// Auth interfaces
export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserResponse {
  id: number;
  email: string;
  name: string | null;
  avatar_url: string | null;
  oauth_provider: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
}

export interface OAuthConfig {
  google_enabled: boolean;
  github_enabled: boolean;
}

// Job interfaces
export interface Job {
  id: number;
  name: string;
  description: string | null;
  collection_name: string;
  status: string;
  document_count: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface JobListResponse {
  total: number;
  jobs: Job[];
}

export interface JobDocument {
  id: number;
  doc_id: string;
  filename: string;
  title: string | null;
  authors: string | null;
  year: number | null;
  phase: string | null;
  topic_category: string | null;
  status: string;
  chunk_count: number;
  total_pages: number | null;
  created_at: string;
}

export interface JobDocumentsResponse {
  total: number;
  documents: JobDocument[];
}

export interface JobStats {
  job_id: number;
  document_count: number;
  chunk_count: number;
  phases: Record<string, number>;
  topics: Record<string, number>;
  year_range: {
    min: number | null;
    max: number | null;
  };
}

export interface JobQueryResult {
  content: string;
  metadata: {
    doc_id: string;
    title?: string;
    authors?: string;
    year?: number;
    phase?: string;
    topic_category?: string;
  };
  score: number;
}

export interface JobQueryResponse {
  question: string;
  results: JobQueryResult[];
  message?: string;
}

export interface JobChatResponse {
  question: string;
  answer: string;
  sources: Array<{
    citation_number: number;
    authors: string;
    year: number | string;
    title: string;
    doc_id: string;
  }>;
  complexity: 'simple' | 'medium' | 'complex';
  pipeline_stats: PipelineStats;
  model: string;
  filters_applied: Record<string, string>;
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

  // Query (for chat) - uses agentic LLM-powered /api/chat endpoint
  async query(request: QueryRequest): Promise<ChatResponse> {
    const searchParams = new URLSearchParams();
    searchParams.set('question', request.question);
    if (request.n_results) searchParams.set('n_sources', request.n_results.toString());
    if (request.phase_filter) searchParams.set('phase_filter', request.phase_filter);
    if (request.topic_filter) searchParams.set('topic_filter', request.topic_filter);
    if (request.deep_analysis) searchParams.set('deep_analysis', 'true');

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

  // Async upload PDF (returns task_id for polling)
  async uploadPDFAsync(
    file: File,
    phase: string,
    topic: string
  ): Promise<AsyncUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('phase', phase);
    formData.append('topic', topic);

    const response = await fetch(`${this.baseUrl}/api/upload/async`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Get upload task status
  async getUploadStatus(taskId: string): Promise<TaskStatusResponse> {
    return this.fetch(`/api/upload/${encodeURIComponent(taskId)}/status`);
  }

  // Poll for upload completion
  async pollUploadStatus(
    taskId: string,
    onProgress?: (status: TaskStatusResponse) => void,
    intervalMs: number = 500,
    maxAttempts: number = 600 // 5 minutes max
  ): Promise<TaskStatusResponse> {
    let attempts = 0;

    while (attempts < maxAttempts) {
      const status = await this.getUploadStatus(taskId);

      if (onProgress) {
        onProgress(status);
      }

      if (status.status === 'completed' || status.status === 'failed') {
        return status;
      }

      await new Promise(resolve => setTimeout(resolve, intervalMs));
      attempts++;
    }

    throw new Error('Upload polling timed out');
  }

  // ============================================
  // Authentication Methods
  // ============================================

  // Register new user
  async register(email: string, password: string, name?: string): Promise<TokenResponse> {
    return this.fetch('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, name }),
    });
  }

  // Login with email/password
  async login(email: string, password: string): Promise<TokenResponse> {
    return this.fetch('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  // Refresh access token
  async refreshToken(refreshToken: string): Promise<TokenResponse> {
    return this.fetch('/api/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  }

  // Logout (invalidate refresh token)
  async logout(refreshToken: string): Promise<{ message: string }> {
    return this.fetch('/api/auth/logout', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  }

  // Get current user (requires auth)
  async getCurrentUser(accessToken: string): Promise<UserResponse> {
    return this.fetch('/api/auth/me', {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
  }

  // Get OAuth configuration
  async getOAuthConfig(): Promise<OAuthConfig> {
    return this.fetch('/api/auth/oauth/config');
  }

  // Get OAuth authorization URL
  async getOAuthUrl(provider: 'google' | 'github'): Promise<{ auth_url: string }> {
    return this.fetch(`/api/auth/oauth/${provider}`);
  }

  // Handle OAuth callback
  async handleOAuthCallback(
    provider: 'google' | 'github',
    code: string,
    state?: string
  ): Promise<TokenResponse> {
    return this.fetch(`/api/auth/oauth/${provider}/callback`, {
      method: 'POST',
      body: JSON.stringify({ code, state }),
    });
  }

  // ============================================
  // Job Methods
  // ============================================

  // Create a new job
  async createJob(
    name: string,
    description?: string,
    accessToken?: string
  ): Promise<Job> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    return this.fetch('/api/jobs', {
      method: 'POST',
      headers,
      body: JSON.stringify({ name, description }),
    });
  }

  // List user's jobs
  async listJobs(accessToken?: string): Promise<JobListResponse> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    return this.fetch('/api/jobs', { headers });
  }

  // Get a specific job
  async getJob(jobId: number, accessToken?: string): Promise<Job> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    return this.fetch(`/api/jobs/${jobId}`, { headers });
  }

  // Update a job
  async updateJob(
    jobId: number,
    updates: { name?: string; description?: string },
    accessToken?: string
  ): Promise<Job> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    return this.fetch(`/api/jobs/${jobId}`, {
      method: 'PATCH',
      headers,
      body: JSON.stringify(updates),
    });
  }

  // Delete a job
  async deleteJob(jobId: number, accessToken?: string): Promise<{ message: string }> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    const response = await fetch(`${this.baseUrl}/api/jobs/${jobId}`, {
      method: 'DELETE',
      headers,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
  }

  // Get job statistics
  async getJobStats(jobId: number, accessToken?: string): Promise<JobStats> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    return this.fetch(`/api/jobs/${jobId}/stats`, { headers });
  }

  // Get job documents
  async getJobDocuments(
    jobId: number,
    params?: { phase_filter?: string; topic_filter?: string; limit?: number },
    accessToken?: string
  ): Promise<JobDocumentsResponse> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    const searchParams = new URLSearchParams();
    if (params?.phase_filter) searchParams.set('phase_filter', params.phase_filter);
    if (params?.topic_filter) searchParams.set('topic_filter', params.topic_filter);
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    return this.fetch(`/api/jobs/${jobId}/documents?${searchParams.toString()}`, { headers });
  }

  // Upload PDF to a job
  async uploadToJob(
    jobId: number,
    file: File,
    phase: string,
    topic: string,
    accessToken?: string
  ): Promise<AsyncUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('phase', phase);
    formData.append('topic', topic);

    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }

    const response = await fetch(`${this.baseUrl}/api/jobs/${jobId}/upload`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Delete document from a job
  async deleteJobDocument(
    jobId: number,
    docId: string,
    accessToken?: string
  ): Promise<{ message: string; chunks_deleted: number }> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    const response = await fetch(
      `${this.baseUrl}/api/jobs/${jobId}/documents/${encodeURIComponent(docId)}`,
      {
        method: 'DELETE',
        headers,
      }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
  }

  // Query a job's knowledge base (raw search results)
  async queryJob(
    jobId: number,
    question: string,
    options?: {
      n_sources?: number;
      phase_filter?: string;
      topic_filter?: string;
    },
    accessToken?: string
  ): Promise<JobQueryResponse> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    const searchParams = new URLSearchParams();
    searchParams.set('question', question);
    if (options?.n_sources) searchParams.set('n_sources', options.n_sources.toString());
    if (options?.phase_filter) searchParams.set('phase_filter', options.phase_filter);
    if (options?.topic_filter) searchParams.set('topic_filter', options.topic_filter);
    return this.fetch(`/api/jobs/${jobId}/query?${searchParams.toString()}`, { headers });
  }

  // Chat with a job's knowledge base (LLM-powered with agentic pipeline)
  async chatJob(
    jobId: number,
    question: string,
    options?: {
      n_sources?: number;
      phase_filter?: string;
      topic_filter?: string;
      deep_analysis?: boolean;
    },
    accessToken?: string
  ): Promise<JobChatResponse> {
    const headers: Record<string, string> = {};
    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }
    const searchParams = new URLSearchParams();
    searchParams.set('question', question);
    if (options?.n_sources) searchParams.set('n_sources', options.n_sources.toString());
    if (options?.phase_filter) searchParams.set('phase_filter', options.phase_filter);
    if (options?.topic_filter) searchParams.set('topic_filter', options.topic_filter);
    if (options?.deep_analysis) searchParams.set('deep_analysis', 'true');
    return this.fetch(`/api/jobs/${jobId}/chat?${searchParams.toString()}`, { headers });
  }
}

export const api = new ApiClient(API_BASE_URL);
