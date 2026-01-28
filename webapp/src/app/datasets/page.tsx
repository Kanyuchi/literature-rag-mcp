"use client";

import { useState } from "react";
import { Database, FileText, Calendar, Tag, Users } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Pagination } from "@/components/layout/pagination";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { usePapers, useCollectionStats } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";

export default function DatasetsPage() {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [phaseFilter, setPhaseFilter] = useState<string | undefined>();
  const [topicFilter, setTopicFilter] = useState<string | undefined>();
  const [searchQuery, setSearchQuery] = useState("");

  const { data: stats } = useCollectionStats();
  const { data, isLoading } = usePapers({
    phase_filter: phaseFilter,
    topic_filter: topicFilter,
    limit: 100,
  });

  // Client-side filtering and pagination
  const filteredPapers = data?.papers.filter((paper) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      paper.title.toLowerCase().includes(query) ||
      paper.authors?.toLowerCase().includes(query) ||
      paper.doc_id.toLowerCase().includes(query)
    );
  }) || [];

  const paginatedPapers = filteredPapers.slice(
    (page - 1) * pageSize,
    page * pageSize
  );

  const phases = stats?.phases ? Object.keys(stats.phases) : [];
  const topics = stats?.topics ? Object.keys(stats.topics) : [];

  return (
    <div>
      <PageHeader
        title="Dataset"
        icon={<Database className="h-6 w-6" />}
        showFilter={false}
        showSearch={true}
        searchPlaceholder="Search papers..."
        searchValue={searchQuery}
        onSearchChange={setSearchQuery}
        actionLabel="Create Dataset"
        onAction={() => {}}
      >
        {/* Filter dropdowns */}
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
      </PageHeader>

      {/* Papers Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <Skeleton key={i} className="h-48 rounded-lg" />
          ))}
        </div>
      ) : paginatedPapers.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {paginatedPapers.map((paper) => (
            <Card
              key={paper.doc_id}
              className="bg-card hover:bg-secondary/30 transition-colors cursor-pointer"
            >
              <CardContent className="p-4 space-y-3">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-primary/10 shrink-0">
                    <FileText className="h-5 w-5 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-medium text-sm leading-tight line-clamp-2">
                      {paper.title}
                    </h3>
                  </div>
                </div>

                {paper.authors && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Users className="h-3 w-3" />
                    <span className="truncate">{paper.authors}</span>
                  </div>
                )}

                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  {paper.year && (
                    <div className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      <span>{paper.year}</span>
                    </div>
                  )}
                  {paper.phase && (
                    <Badge variant="secondary" className="text-xs">
                      {paper.phase}
                    </Badge>
                  )}
                </div>

                {paper.topic && (
                  <div className="flex items-center gap-1">
                    <Tag className="h-3 w-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground truncate">
                      {paper.topic}
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-muted-foreground">
          No papers found
        </div>
      )}

      <Pagination
        total={filteredPapers.length}
        page={page}
        pageSize={pageSize}
        onPageChange={setPage}
        onPageSizeChange={(size) => {
          setPageSize(size);
          setPage(1);
        }}
      />
    </div>
  );
}
