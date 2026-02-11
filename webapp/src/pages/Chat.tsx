import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Send, Loader2, Bot, User, ChevronDown, Filter, Database, Folder, Sparkles, Clock, Zap, ChevronRight, Download, Plus, FolderOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { useStats } from '@/hooks/useApi';
import { useKnowledgeBase } from '@/contexts/KnowledgeBaseContext';
import { useAuth } from '@/contexts/AuthContext';
import { api } from '@/lib/api';
import type { PipelineStats } from '@/lib/api';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

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
  complexity?: 'simple' | 'medium' | 'complex';
  pipelineStats?: PipelineStats;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [phaseFilter, setPhaseFilter] = useState<string>('');
  const [topicFilter, setTopicFilter] = useState<string>('');
  const [synthesisMode, setSynthesisMode] = useState<'paragraph' | 'bullet_points' | 'structured'>('paragraph');
  const [deepAnalysis, setDeepAnalysis] = useState(false);
  const [expandedStats, setExpandedStats] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Array<{ id: number; title?: string | null; created_at: string; updated_at: string }>>([]);
  const [activeSession, setActiveSession] = useState<{ id: number; title?: string | null; created_at: string; updated_at: string } | null>(null);
  const [sessionLoading, setSessionLoading] = useState(false);

  const { selectedKB, isDefaultSelected } = useKnowledgeBase();
  const { accessToken } = useAuth();
  const { data: defaultStats } = useStats(accessToken || undefined);

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
    setMessages([]);
    setActiveSession(null);
  }, [selectedKB?.id]);

  // Load chat sessions for non-default KBs
  useEffect(() => {
    if (!isDefaultSelected && selectedKB && accessToken) {
      setSessionLoading(true);
      api.listChatSessions(selectedKB.id as number, accessToken)
        .then((response) => setSessions(response.sessions))
        .catch((err) => console.error('Failed to load chat sessions:', err))
        .finally(() => setSessionLoading(false));
    } else {
      setSessions([]);
      setActiveSession(null);
    }
  }, [selectedKB, isDefaultSelected, accessToken]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const formatSessionTitle = (session: { title?: string | null; created_at: string; id: number }) => {
    if (session.title && session.title.trim().length > 0) return session.title;
    const created = new Date(session.created_at);
    if (isNaN(created.getTime())) return `Session ${session.id}`;
    const formatted = created.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
    return `Session ${formatted}`;
  };

  const buildSessionTitle = (text: string) => {
    const normalized = text.trim().replace(/\s+/g, ' ');
    if (!normalized) return undefined;
    return normalized.length > 60 ? `${normalized.slice(0, 57)}...` : normalized;
  };

  const hydrateSessionMessages = (storedMessages: Array<{ id: number; role: string; content: string; citations?: Array<Record<string, unknown>> | null }>) => {
    const mapped: Message[] = storedMessages.map((msg) => ({
      id: `session-${msg.id}`,
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content,
      sources: Array.isArray(msg.citations)
        ? msg.citations.map((citation, idx) => {
            const cite = citation as Record<string, unknown>;
            const authors = typeof cite.authors === 'string' ? cite.authors : 'Unknown';
            const year = typeof cite.year === 'string' || typeof cite.year === 'number' ? cite.year : 'n.d.';
            const title = typeof cite.title === 'string' ? cite.title : 'Untitled';
            return {
              title: `[${idx + 1}] ${authors} (${year}). ${title}`,
              score: idx + 1,
            };
          })
        : undefined,
    }));
    setMessages(mapped);
  };

  const handleCreateSession = async () => {
    if (!selectedKB || !accessToken || isDefaultSelected) return;
    try {
      const newSession = await api.createChatSession(selectedKB.id as number, undefined, accessToken);
      setSessions((prev) => [newSession, ...prev]);
      setActiveSession(newSession);
      setMessages([]);
    } catch (err) {
      console.error('Failed to create chat session:', err);
    }
  };

  const handleOpenSession = async (sessionId: number) => {
    if (!accessToken) return;
    try {
      setSessionLoading(true);
      const detail = await api.getChatSession(sessionId, accessToken);
      setActiveSession(detail.session);
      hydrateSessionMessages(detail.messages);
    } catch (err) {
      console.error('Failed to load chat session:', err);
    } finally {
      setSessionLoading(false);
    }
  };

  const handleClearSession = () => {
    setActiveSession(null);
    setMessages([]);
  };

  const handleExportSession = async () => {
    if (!activeSession || !accessToken) return;
    try {
      const blob = await api.exportChatSession(activeSession.id, 'md', accessToken);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `chat-session-${activeSession.id}.md`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export chat session:', err);
    }
  };

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
      let sessionToUse = activeSession;
      if (!isDefaultSelected && selectedKB && accessToken && !sessionToUse) {
        try {
          const newSession = await api.createChatSession(
            selectedKB.id as number,
            buildSessionTitle(userMessage.content),
            accessToken
          );
          setSessions((prev) => [newSession, ...prev]);
          setActiveSession(newSession);
          sessionToUse = newSession;
        } catch (err) {
          console.error('Failed to auto-create chat session:', err);
        }
      }

      if (sessionToUse && accessToken) {
        await api.addChatMessage(
          sessionToUse.id,
          {
            role: 'user',
            content: userMessage.content,
          },
          accessToken
        );
      }

      let assistantContent = '';
      let sources: Message['sources'] = [];
      let complexity: Message['complexity'] = undefined;
      let pipelineStats: Message['pipelineStats'] = undefined;
      let modelUsed: string | undefined = undefined;
      let storedCitations: Array<Record<string, unknown>> | undefined = undefined;

      if (isDefaultSelected) {
        // Query default collection with agentic pipeline
        const response = await api.query({
          question: userMessage.content,
          n_results: 5,
          synthesis_mode: synthesisMode,
          phase_filter: phaseFilter || undefined,
          topic_filter: topicFilter || undefined,
          deep_analysis: deepAnalysis,
        }, accessToken || undefined);

        assistantContent = response.answer;
        sources = response.sources?.map(source => ({
          title: `[${source.citation_number}] ${source.authors} (${source.year}). ${source.title}`,
          score: source.citation_number,
        }));
        complexity = response.complexity;
        pipelineStats = response.pipeline_stats;
        modelUsed = response.model;
        storedCitations = response.sources?.map(source => ({
          citation_number: source.citation_number,
          authors: source.authors,
          year: source.year,
          title: source.title,
          doc_id: source.doc_id,
        }));
      } else {
        // Query job collection with agentic pipeline (same as default)
        const response = await api.chatJob(
          selectedKB.id as number,
          userMessage.content,
          {
            n_sources: 5,
            phase_filter: phaseFilter || undefined,
            topic_filter: topicFilter || undefined,
            deep_analysis: deepAnalysis,
          },
          accessToken || undefined
        );

        assistantContent = response.answer;
        sources = response.sources?.map(source => ({
          title: `[${source.citation_number}] ${source.authors} (${source.year}). ${source.title}`,
          score: source.citation_number,
        }));
        complexity = response.complexity;
        pipelineStats = response.pipeline_stats;
        modelUsed = response.model;
        storedCitations = response.sources?.map(source => ({
          citation_number: source.citation_number,
          authors: source.authors,
          year: source.year,
          title: source.title,
          doc_id: source.doc_id,
        }));
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: assistantContent,
        sources,
        complexity,
        pipelineStats,
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (sessionToUse && accessToken) {
        await api.addChatMessage(
          sessionToUse.id,
          {
            role: 'assistant',
            content: assistantContent,
            citations: storedCitations,
            model: modelUsed,
          },
          accessToken
        );
      }
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
  }, [input, loading, selectedKB, isDefaultSelected, accessToken, phaseFilter, topicFilter, synthesisMode, deepAnalysis, activeSession]);

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
                {!isDefaultSelected && (
                  <span className="text-xs text-muted-foreground/70">History saved only in opened sessions</span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Chat Sessions (only for non-default KBs) */}
            {!isDefaultSelected && selectedKB && accessToken && (
              <>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2">
                      <FolderOpen className="w-4 h-4" />
                      {activeSession ? formatSessionTitle(activeSession) : 'Open Session'}
                      <ChevronDown className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="bg-card border-border">
                    <DropdownMenuItem onClick={handleCreateSession} disabled={sessionLoading}>
                      <Plus className="w-4 h-4 mr-2" />
                      New Session
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={handleClearSession} disabled={!activeSession}>
                      Clear Session
                    </DropdownMenuItem>
                    {sessions.length === 0 && (
                      <DropdownMenuItem disabled>
                        No saved sessions
                      </DropdownMenuItem>
                    )}
                    {sessions.map((session) => (
                      <DropdownMenuItem key={session.id} onClick={() => handleOpenSession(session.id)}>
                        {formatSessionTitle(session)}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
                <Button
                  variant="outline"
                  className="border-border bg-secondary/50 hover:bg-secondary gap-2"
                  onClick={handleExportSession}
                  disabled={!activeSession}
                >
                  <Download className="w-4 h-4" />
                  Export
                </Button>
              </>
            )}

            {/* Deep Analysis Toggle */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border bg-secondary/50">
              <Sparkles className={`w-4 h-4 ${deepAnalysis ? 'text-primary' : 'text-muted-foreground'}`} />
              <Label htmlFor="deep-analysis" className="text-sm cursor-pointer">
                Deep Analysis
              </Label>
              <Switch
                id="deep-analysis"
                checked={deepAnalysis}
                onCheckedChange={setDeepAnalysis}
                className="data-[state=checked]:bg-primary"
              />
            </div>

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

                        {/* Complexity Badge and Pipeline Stats */}
                        {message.role === 'assistant' && message.complexity && (
                          <div className="mt-3 pt-3 border-t border-border/50">
                            <div className="flex items-center gap-2 flex-wrap">
                              {/* Complexity Badge */}
                              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                                message.complexity === 'simple'
                                  ? 'bg-green-500/20 text-green-400'
                                  : message.complexity === 'medium'
                                  ? 'bg-yellow-500/20 text-yellow-400'
                                  : 'bg-purple-500/20 text-purple-400'
                              }`}>
                                {message.complexity === 'complex' && <Sparkles className="w-3 h-3" />}
                                {message.complexity === 'simple' && <Zap className="w-3 h-3" />}
                                {message.complexity.charAt(0).toUpperCase() + message.complexity.slice(1)}
                              </span>

                              {/* Time Badge */}
                              {message.pipelineStats && (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-secondary text-muted-foreground">
                                  <Clock className="w-3 h-3" />
                                  {(message.pipelineStats.total_time_ms / 1000).toFixed(1)}s
                                </span>
                              )}

                              {/* Expandable Stats Button */}
                              {message.pipelineStats && (
                                <button
                                  onClick={() => setExpandedStats(expandedStats === message.id ? null : message.id)}
                                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-secondary/50 hover:bg-secondary text-muted-foreground transition-colors"
                                >
                                  <ChevronRight className={`w-3 h-3 transition-transform ${expandedStats === message.id ? 'rotate-90' : ''}`} />
                                  Details
                                </button>
                              )}
                            </div>

                            {/* Expanded Pipeline Stats */}
                            {expandedStats === message.id && message.pipelineStats && (
                              <div className="mt-2 p-2 rounded bg-secondary/30 text-xs space-y-1">
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                                  <span className="text-muted-foreground">LLM Calls:</span>
                                  <span>{message.pipelineStats.llm_calls}</span>
                                  <span className="text-muted-foreground">Retrieval Attempts:</span>
                                  <span>{message.pipelineStats.retrieval_attempts}</span>
                                  {message.pipelineStats.validation_passed !== null && (
                                    <>
                                      <span className="text-muted-foreground">Validation:</span>
                                      <span className={message.pipelineStats.validation_passed ? 'text-green-400' : 'text-red-400'}>
                                        {message.pipelineStats.validation_passed ? 'Passed' : 'Failed'}
                                      </span>
                                    </>
                                  )}
                                  {(message.pipelineStats.retries.retrieval > 0 || message.pipelineStats.retries.generation > 0) && (
                                    <>
                                      <span className="text-muted-foreground">Retries:</span>
                                      <span>
                                        {message.pipelineStats.retries.retrieval > 0 && `Retrieval: ${message.pipelineStats.retries.retrieval}`}
                                        {message.pipelineStats.retries.retrieval > 0 && message.pipelineStats.retries.generation > 0 && ', '}
                                        {message.pipelineStats.retries.generation > 0 && `Generation: ${message.pipelineStats.retries.generation}`}
                                      </span>
                                    </>
                                  )}
                                </div>
                                {message.pipelineStats.evaluation_scores && (
                                  <div className="mt-2 pt-2 border-t border-border/30">
                                    <span className="text-muted-foreground block mb-1">Evaluation Scores:</span>
                                    <div className="flex gap-3">
                                      <span>Relevance: {(message.pipelineStats.evaluation_scores.relevance * 100).toFixed(0)}%</span>
                                      <span>Coverage: {(message.pipelineStats.evaluation_scores.coverage * 100).toFixed(0)}%</span>
                                      <span>Diversity: {(message.pipelineStats.evaluation_scores.diversity * 100).toFixed(0)}%</span>
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}

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
