import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Send, Loader2, Bot, User, ChevronDown, Filter } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { useQuery, useStats } from '@/hooks/useApi';
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
  const [phaseFilter, setPhaseFilter] = useState<string>('');
  const [topicFilter, setTopicFilter] = useState<string>('');
  const [synthesisMode, setSynthesisMode] = useState<'paragraph' | 'bullet_points' | 'structured'>('paragraph');
  const { query, loading } = useQuery();
  const { data: stats } = useStats();
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const phases = stats ? Object.keys(stats.phases) : [];
  const topics = stats ? Object.keys(stats.topics) : [];

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    try {
      const response = await query({
        question: userMessage.content,
        n_results: 5,
        synthesis_mode: synthesisMode,
        phase_filter: phaseFilter || undefined,
        topic_filter: topicFilter || undefined,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        sources: response.sources?.map(source => ({
          title: `[${source.citation_number}] ${source.authors} (${source.year}). ${source.title}`,
          score: source.citation_number,
        })),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

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
            <h1 className="text-2xl font-semibold text-white">Chat Apps</h1>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Synthesis Mode */}
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

            {/* Phase Filter */}
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

            {/* Topic Filter */}
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
                  <p className="text-sm">Ask questions about the academic literature</p>
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
                  placeholder="Ask a question about the literature..."
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
