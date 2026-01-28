"use client";

import { useState } from "react";
import { Search, FileText, Calendar, Tag, Loader2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Pagination } from "@/components/layout/pagination";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSemanticSearch, useCollectionStats } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import type { SemanticSearchParams } from "@/types/api";

export default function SearchPage() {
  const [searchParams, setSearchParams] = useState<SemanticSearchParams | null>(null);
  const [query, setQuery] = useState("");
  const [phaseFilter, setPhaseFilter] = useState<string | undefined>();
  const [topicFilter, setTopicFilter] = useState<string | undefined>();
  const [nResults, setNResults] = useState(10);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const { data: stats } = useCollectionStats();
  const { data: results, isLoading, isFetching } = useSemanticSearch(searchParams);

  const phases = stats?.phases ? Object.keys(stats.phases) : [];
  const topics = stats?.topics ? Object.keys(stats.topics) : [];

  const handleSearch = () => {
    if (!query.trim()) return;
    setSearchParams({
      query: query.trim(),
      n_results: nResults,
      phase_filter: phaseFilter,
      topic_filter: topicFilter,
    });
    setPage(1);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  // Paginate results
  const paginatedResults = results?.slice(
    (page - 1) * pageSize,
    page * pageSize
  ) || [];

  return (
    <div>
      <PageHeader
        title="Search Apps"
        icon={<Search className="h-6 w-6" />}
        showFilter={false}
        showSearch={false}
        actionLabel="Create Search"
        onAction={() => {}}
      />

      {/* Search Form */}
      <Card className="bg-card mb-6">
        <CardContent className="p-4 space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter your semantic search query..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              className="bg-secondary border-none flex-1"
            />
            <Button onClick={handleSearch} disabled={!query.trim() || isFetching}>
              {isFetching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              <span className="ml-2">Search</span>
            </Button>
          </div>

          <div className="flex flex-wrap gap-3">
            <Select
              value={phaseFilter || "all"}
              onValueChange={(v) => setPhaseFilter(v === "all" ? undefined : v)}
            >
              <SelectTrigger className="w-32 bg-secondary border-none">
                <SelectValue placeholder="Phase" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Phases</SelectItem>
                {phases.map((phase) => (
                  <SelectItem key={phase} value={phase}>
                    {phase}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={topicFilter || "all"}
              onValueChange={(v) => setTopicFilter(v === "all" ? undefined : v)}
            >
              <SelectTrigger className="w-40 bg-secondary border-none">
                <SelectValue placeholder="Topic" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Topics</SelectItem>
                {topics.map((topic) => (
                  <SelectItem key={topic} value={topic}>
                    {topic}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={String(nResults)}
              onValueChange={(v) => setNResults(Number(v))}
            >
              <SelectTrigger className="w-32 bg-secondary border-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="5">5 results</SelectItem>
                <SelectItem value="10">10 results</SelectItem>
                <SelectItem value="20">20 results</SelectItem>
                <SelectItem value="50">50 results</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {isLoading ? (
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-32 rounded-lg" />
          ))}
        </div>
      ) : results && results.length > 0 ? (
        <>
          <div className="space-y-4">
            {paginatedResults.map((result, idx) => (
              <Card key={`${result.doc_id}-${idx}`} className="bg-card hover:bg-secondary/30 transition-colors">
                <CardContent className="p-4 space-y-3">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-lg bg-primary/10 shrink-0">
                        <FileText className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-medium">{result.title}</h3>
                        {result.authors && (
                          <p className="text-sm text-muted-foreground mt-1">
                            {result.authors}
                          </p>
                        )}
                      </div>
                    </div>
                    <Badge variant="outline" className="shrink-0">
                      {(result.relevance_score * 100).toFixed(1)}% match
                    </Badge>
                  </div>

                  <p className="text-sm text-muted-foreground line-clamp-3 bg-secondary/50 p-3 rounded">
                    {result.chunk_text}
                  </p>

                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    {result.year && (
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        <span>{result.year}</span>
                      </div>
                    )}
                    {result.phase && (
                      <Badge variant="secondary" className="text-xs">
                        {result.phase}
                      </Badge>
                    )}
                    {result.topic && (
                      <div className="flex items-center gap-1">
                        <Tag className="h-3 w-3" />
                        <span>{result.topic}</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Pagination
            total={results.length}
            page={page}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size);
              setPage(1);
            }}
          />
        </>
      ) : searchParams ? (
        <div className="text-center py-12 text-muted-foreground">
          No results found for &quot;{searchParams.query}&quot;
        </div>
      ) : (
        <div className="text-center py-12 text-muted-foreground">
          <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Enter a search query to find relevant papers</p>
        </div>
      )}
    </div>
  );
}
