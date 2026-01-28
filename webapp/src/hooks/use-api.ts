"use client";

import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { SemanticSearchParams } from "@/types/api";

// Collection Stats
export function useCollectionStats() {
  return useQuery({
    queryKey: ["collectionStats"],
    queryFn: () => api.getCollectionStats(),
  });
}

// List Papers
export function usePapers(params?: {
  phase_filter?: string;
  topic_filter?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ["papers", params],
    queryFn: () => api.listPapers(params),
  });
}

// Semantic Search
export function useSemanticSearch(params: SemanticSearchParams | null) {
  return useQuery({
    queryKey: ["search", params],
    queryFn: () => (params ? api.semanticSearch(params) : Promise.resolve([])),
    enabled: !!params?.query,
  });
}

// Answer with Citations (as mutation since it's triggered by user action)
export function useAnswerWithCitations() {
  return useMutation({
    mutationFn: (params: {
      question: string;
      n_sources?: number;
      phase_filter?: string;
      topic_filter?: string;
    }) => api.answerWithCitations(params),
  });
}

// Find Related Papers
export function useRelatedPapers(paperId: string | null, n_results?: number) {
  return useQuery({
    queryKey: ["relatedPapers", paperId, n_results],
    queryFn: () =>
      paperId
        ? api.findRelatedPapers({ paper_id: paperId, n_results })
        : Promise.resolve({ source_paper_id: "", related_papers: [] }),
    enabled: !!paperId,
  });
}

// Synthesis Query (as mutation)
export function useSynthesisQuery() {
  return useMutation({
    mutationFn: (params: {
      question: string;
      topics: string[];
      n_per_topic?: number;
    }) => api.synthesisQuery(params),
  });
}
