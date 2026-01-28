// API Response Types based on MCP server tools

export interface Paper {
  doc_id: string;
  title: string;
  authors?: string;
  year?: number;
  phase?: string;
  topic?: string;
  source?: string;
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

export interface CollectionStats {
  total_papers: number;
  total_chunks: number;
  phases: Record<string, number>;
  topics: Record<string, number>;
  year_range: {
    min: number;
    max: number;
  };
}

export interface Citation {
  doc_id: string;
  title: string;
  authors?: string;
  year?: number;
  chunk_text: string;
}

export interface AnswerWithCitations {
  sources: Citation[];
  bibliography: string[];
  suggested_structure: string[];
}

export interface SynthesisResult {
  [topic: string]: string;
}

export interface ListPapersResponse {
  total: number;
  papers: Paper[];
}

export interface SemanticSearchParams {
  query: string;
  n_results?: number;
  phase_filter?: string;
  topic_filter?: string;
  year_min?: number;
  year_max?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  name: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}
