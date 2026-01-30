import { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Filter, Search, ChevronLeft, ChevronRight, FileText, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';
import { usePapers, useStats } from '@/hooks/useApi';

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

export default function Dataset() {
  const [searchQuery, setSearchQuery] = useState('');
  const [phaseFilter, setPhaseFilter] = useState<string>('');
  const [topicFilter, setTopicFilter] = useState<string>('');
  
  const { data: papersData, loading, error } = usePapers({ limit: 100 });
  const { data: stats } = useStats();

  // Debug logging
  console.log('Papers data:', papersData);
  console.log('Loading:', loading);
  console.log('Error:', error);

  // Filter papers based on search and filters
  const filteredPapers = papersData?.papers?.filter(paper => {
    const matchesSearch = !searchQuery || 
      paper.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      paper.authors?.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesPhase = !phaseFilter || paper.phase === phaseFilter;
    const matchesTopic = !topicFilter || paper.topic === topicFilter;
    
    return matchesSearch && matchesPhase && matchesTopic;
  }) || [];

  const phases = stats ? Object.keys(stats.phases) : [];
  const topics = stats ? Object.keys(stats.topics) : [];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-[calc(100vh-72px)] bg-background px-4 md:px-8 lg:px-12 py-6"
    >
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div 
          variants={itemVariants}
          className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <Database className="w-5 h-5 text-primary" />
            </div>
            <h1 className="text-2xl font-semibold text-white">Dataset</h1>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Phase Filter */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2">
                  <Filter className="w-4 h-4" />
                  {phaseFilter || 'Phase'}
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
            
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input 
                placeholder="Search papers..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-[200px] bg-secondary/50 border-border focus:border-primary"
              />
            </div>
          </div>
        </motion.div>

        {/* Debug Info - remove after testing */}
        <div className="mb-4 p-4 bg-secondary/30 rounded-lg text-sm">
          <p>Loading: {loading ? 'true' : 'false'}</p>
          <p>Error: {error || 'none'}</p>
          <p>Papers data exists: {papersData ? 'yes' : 'no'}</p>
          <p>Papers array exists: {papersData?.papers ? 'yes' : 'no'}</p>
          <p>Papers count: {papersData?.papers?.length ?? 'N/A'}</p>
          <p>Filtered count: {filteredPapers.length}</p>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-32">
            <Loader2 className="w-8 h-8 animate-spin text-primary mb-4" />
            <p className="text-muted-foreground">Loading papers...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="flex flex-col items-center justify-center py-32">
            <p className="text-destructive mb-2">Failed to load papers</p>
            <p className="text-muted-foreground text-sm">{error}</p>
          </div>
        )}

        {/* Papers Table */}
        {!loading && !error && (
          <motion.div variants={itemVariants}>
            <div className="rounded-lg border border-border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow className="bg-secondary/30 hover:bg-secondary/30 border-border">
                    <TableHead className="text-muted-foreground font-medium">Paper</TableHead>
                    <TableHead className="text-muted-foreground font-medium">Authors</TableHead>
                    <TableHead className="text-muted-foreground font-medium">Year</TableHead>
                    <TableHead className="text-muted-foreground font-medium">Phase</TableHead>
                    <TableHead className="text-muted-foreground font-medium">Topic</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredPapers.length === 0 ? (
                    <TableRow className="border-border">
                      <TableCell colSpan={5} className="text-center py-12 text-muted-foreground">
                        No papers found
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredPapers.map((paper) => (
                      <TableRow 
                        key={paper.doc_id} 
                        className="border-border hover:bg-secondary/20 transition-colors"
                      >
                        <TableCell>
                          <div className="flex items-center gap-3">
                            <FileText className="w-5 h-5 text-primary" />
                            <span className="text-foreground font-medium">{paper.title}</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-muted-foreground max-w-[200px] truncate">
                          {paper.authors || 'Unknown'}
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {paper.year || '-'}
                        </TableCell>
                        <TableCell>
                          {paper.phase && (
                            <Badge variant="secondary" className="bg-secondary/50">
                              {paper.phase}
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          {paper.topic && (
                            <Badge variant="secondary" className="bg-secondary/50">
                              {paper.topic}
                            </Badge>
                          )}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </motion.div>
        )}

        {/* Pagination */}
        <motion.div 
          variants={itemVariants}
          className="flex items-center justify-end gap-4 mt-8 pt-4 border-t border-border"
        >
          <span className="text-sm text-muted-foreground">
            Total {filteredPapers.length}
          </span>
          
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" className="border-border bg-secondary/50 hover:bg-secondary" disabled>
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="icon" className="border-border bg-secondary/50 hover:bg-secondary" disabled>
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2">
                50 / Page <ChevronLeft className="w-4 h-4 rotate-90" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="bg-card border-border">
              <DropdownMenuItem>10 / Page</DropdownMenuItem>
              <DropdownMenuItem>20 / Page</DropdownMenuItem>
              <DropdownMenuItem>50 / Page</DropdownMenuItem>
              <DropdownMenuItem>100 / Page</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </motion.div>
      </div>
    </motion.div>
  );
}
