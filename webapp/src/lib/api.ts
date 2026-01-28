import type {
  CollectionStats,
  ListPapersResponse,
  SearchResult,
  AnswerWithCitations,
  SynthesisResult,
  Paper,
  SemanticSearchParams,
} from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Collection Stats
  async getCollectionStats(): Promise<CollectionStats> {
    return this.fetch<CollectionStats>("/api/stats");
  }

  // List Papers
  async listPapers(params?: {
    phase_filter?: string;
    topic_filter?: string;
    limit?: number;
  }): Promise<ListPapersResponse> {
    const searchParams = new URLSearchParams();
    if (params?.phase_filter) searchParams.set("phase_filter", params.phase_filter);
    if (params?.topic_filter) searchParams.set("topic_filter", params.topic_filter);
    if (params?.limit) searchParams.set("limit", String(params.limit));

    const query = searchParams.toString();
    return this.fetch<ListPapersResponse>(`/api/papers${query ? `?${query}` : ""}`);
  }

  // Semantic Search
  async semanticSearch(params: SemanticSearchParams): Promise<SearchResult[]> {
    const searchParams = new URLSearchParams();
    searchParams.set("query", params.query);
    if (params.n_results) searchParams.set("n_results", String(params.n_results));
    if (params.phase_filter) searchParams.set("phase_filter", params.phase_filter);
    if (params.topic_filter) searchParams.set("topic_filter", params.topic_filter);
    if (params.year_min) searchParams.set("year_min", String(params.year_min));
    if (params.year_max) searchParams.set("year_max", String(params.year_max));

    return this.fetch<SearchResult[]>(`/api/search?${searchParams.toString()}`);
  }

  // Get Context for LLM
  async getContextForLLM(params: {
    query: string;
    n_results?: number;
    phase_filter?: string;
    topic_filter?: string;
  }): Promise<string> {
    const searchParams = new URLSearchParams();
    searchParams.set("query", params.query);
    if (params.n_results) searchParams.set("n_results", String(params.n_results));
    if (params.phase_filter) searchParams.set("phase_filter", params.phase_filter);
    if (params.topic_filter) searchParams.set("topic_filter", params.topic_filter);

    return this.fetch<string>(`/api/context?${searchParams.toString()}`);
  }

  // Answer with Citations
  async answerWithCitations(params: {
    question: string;
    n_sources?: number;
    phase_filter?: string;
    topic_filter?: string;
  }): Promise<AnswerWithCitations> {
    const searchParams = new URLSearchParams();
    searchParams.set("question", params.question);
    if (params.n_sources) searchParams.set("n_sources", String(params.n_sources));
    if (params.phase_filter) searchParams.set("phase_filter", params.phase_filter);
    if (params.topic_filter) searchParams.set("topic_filter", params.topic_filter);

    return this.fetch<AnswerWithCitations>(`/api/answer?${searchParams.toString()}`);
  }

  // Find Related Papers
  async findRelatedPapers(params: {
    paper_id: string;
    n_results?: number;
  }): Promise<{ source_paper_id: string; related_papers: Paper[] }> {
    const searchParams = new URLSearchParams();
    searchParams.set("paper_id", params.paper_id);
    if (params.n_results) searchParams.set("n_results", String(params.n_results));

    return this.fetch<{ source_paper_id: string; related_papers: Paper[] }>(
      `/api/related?${searchParams.toString()}`
    );
  }

  // Synthesis Query
  async synthesisQuery(params: {
    question: string;
    topics: string[];
    n_per_topic?: number;
  }): Promise<SynthesisResult> {
    return this.fetch<SynthesisResult>("/api/synthesis", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }
}

export const api = new ApiClient(API_BASE_URL);
