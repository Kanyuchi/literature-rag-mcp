"use client";

import { useState } from "react";
import { Bot, Plus, Loader2, CheckCircle2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useSynthesisQuery, useCollectionStats } from "@/hooks/use-api";
import { cn } from "@/lib/utils";

export default function AgentsPage() {
  const [showAgent, setShowAgent] = useState(false);
  const [question, setQuestion] = useState("");
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [results, setResults] = useState<Record<string, string> | null>(null);

  const { data: stats } = useCollectionStats();
  const synthesisMutation = useSynthesisQuery();

  const topics = stats?.topics ? Object.keys(stats.topics) : [];

  const toggleTopic = (topic: string) => {
    setSelectedTopics((prev) =>
      prev.includes(topic)
        ? prev.filter((t) => t !== topic)
        : [...prev, topic]
    );
  };

  const handleRun = async () => {
    if (!question.trim() || selectedTopics.length === 0) return;

    try {
      const result = await synthesisMutation.mutateAsync({
        question: question.trim(),
        topics: selectedTopics,
        n_per_topic: 3,
      });
      setResults(result);
    } catch (error) {
      console.error("Synthesis failed:", error);
    }
  };

  if (!showAgent) {
    return (
      <div>
        <PageHeader
          title="Agents"
          icon={<Bot className="h-6 w-6" />}
          showFilter={true}
          showSearch={true}
          searchPlaceholder="Search agents..."
          actionLabel="Create agent"
          onAction={() => setShowAgent(true)}
        />

        {/* Empty state */}
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="p-4 rounded-full bg-primary/10 mb-4">
            <Bot className="h-8 w-8 text-primary" />
          </div>
          <h3 className="text-lg font-medium mb-2">No agents configured</h3>
          <p className="text-muted-foreground mb-4">
            Create a synthesis agent to analyze multiple topics
          </p>
          <Button onClick={() => setShowAgent(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Create agent
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Synthesis Agent"
        icon={<Bot className="h-6 w-6" />}
        showFilter={false}
        showSearch={false}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration */}
        <Card className="bg-card lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">
                Research Question
              </label>
              <Input
                placeholder="Enter your research question..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="bg-secondary border-none"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">
                Select Topics ({selectedTopics.length} selected)
              </label>
              <ScrollArea className="h-64 rounded-md border border-border p-2">
                <div className="space-y-2">
                  {topics.map((topic) => (
                    <button
                      key={topic}
                      onClick={() => toggleTopic(topic)}
                      className={cn(
                        "w-full text-left px-3 py-2 rounded-md text-sm transition-colors",
                        selectedTopics.includes(topic)
                          ? "bg-primary text-primary-foreground"
                          : "bg-secondary hover:bg-secondary/80"
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <span>{topic}</span>
                        {selectedTopics.includes(topic) && (
                          <CheckCircle2 className="h-4 w-4" />
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </ScrollArea>
            </div>

            <Button
              onClick={handleRun}
              disabled={!question.trim() || selectedTopics.length === 0 || synthesisMutation.isPending}
              className="w-full"
            >
              {synthesisMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running Synthesis...
                </>
              ) : (
                <>
                  <Bot className="h-4 w-4 mr-2" />
                  Run Synthesis
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results */}
        <Card className="bg-card lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-lg">Results</CardTitle>
          </CardHeader>
          <CardContent>
            {synthesisMutation.isPending ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <span className="ml-3 text-muted-foreground">
                  Analyzing {selectedTopics.length} topics...
                </span>
              </div>
            ) : results ? (
              <ScrollArea className="h-[500px]">
                <div className="space-y-6">
                  {Object.entries(results).map(([topic, content]) => (
                    <div key={topic} className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">{topic}</Badge>
                      </div>
                      <div className="bg-secondary/50 rounded-lg p-4">
                        <p className="text-sm whitespace-pre-wrap">{content}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Configure and run the synthesis agent</p>
                <p className="text-sm mt-2">
                  Select topics to compare and analyze
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
