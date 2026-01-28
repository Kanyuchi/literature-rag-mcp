"use client";

import { useState } from "react";
import { MessageSquare, Plus, Send, BookOpen, Loader2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAnswerWithCitations } from "@/hooks/use-api";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Array<{
    title: string;
    authors?: string;
    year?: number;
    chunk_text: string;
  }>;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [showChat, setShowChat] = useState(false);

  const answerMutation = useAnswerWithCitations();

  const handleSend = async () => {
    if (!input.trim() || answerMutation.isPending) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setShowChat(true);

    try {
      const result = await answerMutation.mutateAsync({
        question: input,
        n_sources: 5,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: result.suggested_structure.join("\n\n"),
        citations: result.sources.map((s) => ({
          title: s.title,
          authors: s.authors,
          year: s.year,
          chunk_text: s.chunk_text,
        })),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error while processing your question. Please make sure the API server is running.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!showChat) {
    return (
      <div>
        <PageHeader
          title="Chat Apps"
          icon={<MessageSquare className="h-6 w-6" />}
          showFilter={true}
          showSearch={true}
          searchPlaceholder="Search chats..."
          actionLabel="Create chat"
          onAction={() => setShowChat(true)}
        />

        {/* Empty state */}
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="p-4 rounded-full bg-primary/10 mb-4">
            <MessageSquare className="h-8 w-8 text-primary" />
          </div>
          <h3 className="text-lg font-medium mb-2">No chat sessions yet</h3>
          <p className="text-muted-foreground mb-4">
            Start a new chat to ask questions about the literature
          </p>
          <Button onClick={() => setShowChat(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Create chat
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-12rem)] flex flex-col">
      <PageHeader
        title="Chat"
        icon={<MessageSquare className="h-6 w-6" />}
        showFilter={false}
        showSearch={false}
      />

      {/* Chat Messages */}
      <Card className="flex-1 bg-card flex flex-col overflow-hidden">
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <BookOpen className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Ask a question about the literature collection</p>
                <p className="text-sm mt-2">
                  Your answers will include citations from relevant papers
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex",
                    message.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  <div
                    className={cn(
                      "max-w-[80%] rounded-lg p-4",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary"
                    )}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>

                    {/* Citations */}
                    {message.citations && message.citations.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-border/50">
                        <p className="text-xs font-medium mb-2 opacity-70">
                          Sources ({message.citations.length})
                        </p>
                        <div className="space-y-2">
                          {message.citations.map((citation, idx) => (
                            <div
                              key={idx}
                              className="text-xs p-2 rounded bg-background/50"
                            >
                              <p className="font-medium">{citation.title}</p>
                              {citation.authors && (
                                <p className="opacity-70">
                                  {citation.authors}
                                  {citation.year && ` (${citation.year})`}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {answerMutation.isPending && (
              <div className="flex justify-start">
                <div className="bg-secondary rounded-lg p-4 flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm">Searching literature...</span>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Input */}
        <CardContent className="border-t border-border p-4">
          <div className="flex gap-2">
            <Input
              placeholder="Ask a question about the literature..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={answerMutation.isPending}
              className="bg-secondary border-none"
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || answerMutation.isPending}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
