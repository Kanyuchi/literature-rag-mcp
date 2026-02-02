import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Send, Loader2, Bot, User, ChevronDown, Filter, Database, Folder } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { useStats } from '@/hooks/useApi';
import { useKnowledgeBase, DEFAULT_COLLECTION_ID } from '@/contexts/KnowledgeBaseContext';
import { useAuth } from '@/contexts/AuthContext';
import { api } from '@/lib/api';
import type { JobQueryResponse } from '@/lib/api';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ScrollArea } from '@/components/ui/scroll-area';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
    },
  },
};

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    title: string;
    score: number;
  }>;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [phaseFilter, setPhaseFilter] = useState<string>('');
  const [topicFilter, setTopicFilter] = useState<string>('');
  const [synthesisMode, setSynthesisMode] = useState<'paragraph' | 'bullet_points' | 'structured'>('paragraph');

  const { selectedKB, isDefaultSelected } = useKnowledgeBase();
  const { accessToken } = useAuth();
  const { data: defaultStats } = useStats();

  // Job stats for non-default KB
  const [jobStats, setJobStats] = useState<{ phases: Record<string, number>; topics: Record<string, number> } | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load job stats when KB changes
  useEffect(() => {
    if (!isDefaultSelected && selectedKB && accessToken) {
      api.getJobStats(selectedKB.id as number, accessToken)
        .then(stats => setJobStats({ phases: stats.phases, topics: stats.topics }))
        .catch(err => console.error('Failed to load job stats:', err));
    } else {
      setJobStats(null);
    }
  }, [selectedKB, isDefaultSelected, accessToken]);

  // Use appropriate stats based on selected KB
  const stats = isDefaultSelected ? defaultStats : jobStats;
  const phases = stats ? Object.keys(stats.phases) : [];
  const topics = stats ? Object.keys(stats.topics) : [];

  // Reset filters when KB changes
  useEffect(() => {
    setPhaseFilter('');
    setTopicFilter('');
  }, [selectedKB?.id]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading || !selectedKB) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      let assistantContent = '';
      let sources: Message['sources'] = [];

      if (isDefaultSelected) {
        // Query default collection
        const response = await api.query({
          question: userMessage.content,
          n_results: 5,
          synthesis_mode: synthesisMode,
          phase_filter: phaseFilter || undefined,
          topic_filter: topicFilter || undefined,
        });

        assistantContent = response.answer;
        sources = response.sources?.map(source => ({
          title: `[${source.citation_number}] ${source.authors} (${source.year}). ${source.title}`,
          score: source.citation_number,
        }));
      } else {
        // Query job collection
        const response: JobQueryResponse = await api.queryJob(
          selectedKB.id as number,
          userMessage.content,
          {
            n_sources: 5,
            phase_filter: phaseFilter || undefined,
            topic_filter: topicFilter || undefined,
          },
          accessToken || undefined
        );

        if (response.message) {
          assistantContent = response.message;
        } else if (response.results && response.results.length > 0) {
          // Format results into a readable response
          const resultTexts = response.results.map((r, i) => {
            const meta = r.metadata;
            return `**[${i + 1}] ${meta.title || 'Untitled'}** (${meta.authors || 'Unknown'}, ${meta.year || 'n.d.'})\n${r.content.substring(0, 500)}...`;
          });
          assistantContent = `Based on your knowledge base, here are the most relevant passages:\n\n${resultTexts.join('\n\n---\n\n')}`;
          sources = response.results.map((r, i) => ({
            title: `[${i + 1}] ${r.metadata.title || 'Untitled'} - Score: ${r.score.toFixed(3)}`,
            score: r.score,
          }));
        } else {
          assistantContent = 'No relevant documents found for your query.';
        }
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: assistantContent,
        sources,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${err instanceof Error ? err.message : 'Failed to process your request. Please try again.'}`,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, selectedKB, isDefaultSelected, accessToken, phaseFilter, topicFilter, synthesisMode]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-[calc(100vh-72px)] bg-background px-4 md:px-8 lg:px-12 py-6"
    >
      <div className="max-w-[1400px] mx-auto h-[calc(100vh-120px)] flex flex-col">
        {/* Header */}
        <motion.div
          variants={itemVariants}
          className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <MessageSquare className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-white">Chat</h1>
              {/* Show which KB is being queried */}
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                {isDefaultSelected ? (
                  <Database className="w-3 h-3" />
                ) : (
                  <Folder className="w-3 h-3" />
                )}
                <span>Querying: {selectedKB?.name}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Synthesis Mode - only for default collection */}
            {isDefaultSelected && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2">
                    <Filter className="w-4 h-4" />
                    {synthesisMode === 'paragraph' ? 'Paragraph' : synthesisMode === 'bullet_points' ? 'Bullet Points' : 'Structured'}
                    <ChevronDown className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="bg-card border-border">
                  <DropdownMenuItem onClick={() => setSynthesisMode('paragraph')}>Paragraph</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setSynthesisMode('bullet_points')}>Bullet Points</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setSynthesisMode('structured')}>Structured</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}

            {/* Phase Filter */}
            {phases.length > 0 && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2">
                    <Filter className="w-4 h-4" />
                    {phaseFilter || 'Phase'}
                    <ChevronDown className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="bg-card border-border">
                  <DropdownMenuItem onClick={() => setPhaseFilter('')}>All Phases</DropdownMenuItem>
                  {phases.map(phase => (
                    <DropdownMenuItem key={phase} onClick={() => setPhaseFilter(phase)}>
                      {phase}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            )}

            {/* Topic Filter */}
            {topics.length > 0 && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2">
                    <Filter className="w-4 h-4" />
                    {topicFilter || 'Topic'}
                    <ChevronDown className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="bg-card border-border">
                  <DropdownMenuItem onClick={() => setTopicFilter('')}>All Topics</DropdownMenuItem>
                  {topics.map(topic => (
                    <DropdownMenuItem key={topic} onClick={() => setTopicFilter(topic)}>
                      {topic}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        </motion.div>

        {/* Chat Messages */}
        <motion.div variants={itemVariants} className="flex-1 min-h-0">
          <Card className="h-full flex flex-col bg-card border-border">
            <ScrollArea ref={scrollRef} className="flex-1 p-4">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
                  <Bot className="w-16 h-16 mb-4 opacity-50" />
                  <p className="text-lg font-medium">Start a conversation</p>
                  <p className="text-sm text-center max-w-md mt-2">
                    {isDefaultSelected
                      ? 'Ask questions about the academic literature in the demo collection'
                      : `Ask questions about documents in "${selectedKB?.name}"`}
                  </p>
                  {selectedKB && (
                    <p className="text-xs mt-4 text-muted-foreground/70">
                      {selectedKB.document_count} documents · {selectedKB.chunk_count.toLocaleString()} chunks
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.role === 'user'
                          ? 'bg-primary'
                          : 'bg-secondary'
                      }`}>
                        {message.role === 'user' ? (
                          <User className="w-4 h-4 text-primary-foreground" />
                        ) : (
                          <Bot className="w-4 h-4 text-foreground" />
                        )}
                      </div>

                      <div className={`flex-1 max-w-[80%] ${
                        message.role === 'user' ? 'text-right' : ''
                      }`}>
                        <Card className={`inline-block p-4 ${
                          message.role === 'user'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-secondary/50'
                        }`}>
                          <p className="whitespace-pre-wrap text-left">{message.content}</p>
                        </Card>

                        {message.sources && message.sources.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-border/50">
                            <span className="text-xs text-muted-foreground block mb-2">References:</span>
                            <div className="space-y-1">
                              {message.sources.map((source, idx) => (
                                <p key={idx} className="text-xs text-muted-foreground">
                                  {source.title}
                                </p>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {loading && (
                    <div className="flex gap-4">
                      <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                        <Bot className="w-4 h-4" />
                      </div>
                      <Card className="inline-block p-4 bg-secondary/50">
                        <Loader2 className="w-5 h-5 animate-spin" />
                      </Card>
                    </div>
                  )}
                </div>
              )}
            </ScrollArea>

            {/* Input Area */}
            <div className="p-4 border-t border-border">
              <div className="flex gap-2">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={`Ask a question about ${isDefaultSelected ? 'the literature' : selectedKB?.name}...`}
                  className="flex-1 bg-secondary/50 border-border focus:border-primary"
                  disabled={loading}
                />
                <Button
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                  className="bg-white text-background hover:bg-white/90"
                >
                  {loading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
}
