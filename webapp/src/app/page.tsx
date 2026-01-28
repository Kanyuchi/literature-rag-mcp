"use client";

import Link from "next/link";
import { Database, MessageSquare, Search, Bot, ChevronRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useCollectionStats } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";

export default function HomePage() {
  const { data: stats, isLoading } = useCollectionStats();

  return (
    <div className="space-y-12">
      {/* Welcome Banner */}
      <div className="pt-8">
        <h1 className="text-4xl font-semibold">
          Welcome to <span className="text-primary">LitRAG</span>
        </h1>
        {stats && (
          <p className="mt-2 text-muted-foreground">
            {stats.total_papers} papers | {stats.total_chunks.toLocaleString()} indexed chunks
          </p>
        )}
      </div>

      {/* Dataset Section */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <Database className="h-5 w-5 text-muted-foreground" />
          <h2 className="text-xl font-semibold">Dataset</h2>
        </div>

        <Link href="/datasets">
          <Card className="bg-card hover:bg-secondary/50 transition-colors cursor-pointer w-fit">
            <CardContent className="flex items-center gap-2 py-4 px-6">
              <span className="text-sm">See All</span>
              <ChevronRight className="h-4 w-4" />
            </CardContent>
          </Card>
        </Link>

        {/* Stats Cards */}
        {isLoading ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-24 rounded-lg" />
            ))}
          </div>
        ) : stats ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <Card className="bg-card">
              <CardContent className="pt-4">
                <div className="text-2xl font-bold text-primary">{stats.total_papers}</div>
                <div className="text-sm text-muted-foreground">Total Papers</div>
              </CardContent>
            </Card>
            <Card className="bg-card">
              <CardContent className="pt-4">
                <div className="text-2xl font-bold text-primary">
                  {stats.total_chunks.toLocaleString()}
                </div>
                <div className="text-sm text-muted-foreground">Indexed Chunks</div>
              </CardContent>
            </Card>
            <Card className="bg-card">
              <CardContent className="pt-4">
                <div className="text-2xl font-bold text-primary">
                  {Object.keys(stats.phases).length}
                </div>
                <div className="text-sm text-muted-foreground">Phases</div>
              </CardContent>
            </Card>
            <Card className="bg-card">
              <CardContent className="pt-4">
                <div className="text-2xl font-bold text-primary">
                  {Object.keys(stats.topics).length}
                </div>
                <div className="text-sm text-muted-foreground">Topics</div>
              </CardContent>
            </Card>
          </div>
        ) : null}
      </section>

      {/* Chat Apps Section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-muted-foreground" />
            <h2 className="text-xl font-semibold">Apps</h2>
          </div>

          <Tabs defaultValue="chat" className="w-auto">
            <TabsList className="bg-secondary">
              <TabsTrigger value="chat" className="data-[state=active]:bg-card">
                Chat Apps
              </TabsTrigger>
              <TabsTrigger value="search" className="data-[state=active]:bg-card">
                Search Apps
              </TabsTrigger>
              <TabsTrigger value="agent" className="data-[state=active]:bg-card">
                Agent
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        <Link href="/chat">
          <Card className="bg-card hover:bg-secondary/50 transition-colors cursor-pointer w-fit">
            <CardContent className="flex items-center gap-2 py-4 px-6">
              <span className="text-sm">See All</span>
              <ChevronRight className="h-4 w-4" />
            </CardContent>
          </Card>
        </Link>
      </section>

      {/* Quick Actions */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link href="/search">
            <Card className="bg-card hover:bg-secondary/50 transition-colors cursor-pointer h-full">
              <CardContent className="flex items-center gap-4 py-6">
                <div className="p-3 rounded-lg bg-primary/10">
                  <Search className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <div className="font-medium">Semantic Search</div>
                  <div className="text-sm text-muted-foreground">
                    Search across all papers
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>

          <Link href="/chat">
            <Card className="bg-card hover:bg-secondary/50 transition-colors cursor-pointer h-full">
              <CardContent className="flex items-center gap-4 py-6">
                <div className="p-3 rounded-lg bg-primary/10">
                  <MessageSquare className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <div className="font-medium">Ask Questions</div>
                  <div className="text-sm text-muted-foreground">
                    Get answers with citations
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>

          <Link href="/agents">
            <Card className="bg-card hover:bg-secondary/50 transition-colors cursor-pointer h-full">
              <CardContent className="flex items-center gap-4 py-6">
                <div className="p-3 rounded-lg bg-primary/10">
                  <Bot className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <div className="font-medium">Synthesis Agent</div>
                  <div className="text-sm text-muted-foreground">
                    Multi-topic analysis
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        </div>
      </section>
    </div>
  );
}
