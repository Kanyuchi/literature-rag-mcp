import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Database, MessageSquare, ChevronRight, Loader2, FileText, Layers, Calendar, Folder, Plus, Briefcase } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { useStats } from '@/hooks/useApi';
import { Badge } from '@/components/ui/badge';
import { useAuth } from '@/contexts/AuthContext';
import { api } from '@/lib/api';
import type { Job } from '@/lib/api';
import { useTranslation } from 'react-i18next';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
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
      duration: 0.6,
    },
  },
};

export default function Home() {
  const { isAuthenticated, isLoading: authLoading, accessToken, user } = useAuth();
  const { data: stats, loading, error } = useStats(accessToken || undefined);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const { t } = useTranslation();

  // Load user's jobs if authenticated
  const loadJobs = useCallback(async () => {
    if (!accessToken) return;
    setJobsLoading(true);
    try {
      const response = await api.listJobs(accessToken);
      setJobs(response.jobs);
    } catch (err) {
      console.error('Failed to load jobs:', err);
    } finally {
      setJobsLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    if (isAuthenticated && accessToken) {
      loadJobs();
    }
  }, [isAuthenticated, accessToken, loadJobs]);

  // Calculate total stats from user's jobs
  const userTotalDocs = jobs.reduce((sum, job) => sum + job.document_count, 0);
  const userTotalChunks = jobs.reduce((sum, job) => sum + job.chunk_count, 0);

  const isUserLoading = authLoading || (isAuthenticated && jobsLoading);

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-[calc(100vh-72px)] bg-background px-4 md:px-8 lg:px-12 py-8 md:py-12"
    >
      <div className="max-w-[1400px] mx-auto">
        {/* Hero Section */}
        <motion.section variants={itemVariants} className="mb-16">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white">
            {t('home.welcome')} <span className="gradient-text">Retrievo</span>
          </h1>
          <p className="mt-4 text-muted-foreground text-lg">
            {isAuthenticated
              ? t('home.hello', { name: user?.name || user?.email?.split('@')[0] || 'Researcher' })
              : t('home.guest_subtitle')}
          </p>
        </motion.section>

        {/* Show different content based on auth state */}
        {isUserLoading ? (
          <motion.section variants={itemVariants}>
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
          </motion.section>
        ) : isAuthenticated ? (
          <>
            {/* User's Jobs Section */}
            <motion.section variants={itemVariants} className="mb-12">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Briefcase className="w-5 h-5 text-primary" />
                  </div>
                  <h2 className="text-2xl font-semibold text-white">{t('home.your_kbs')}</h2>
                </div>
                <Link
                  to="/jobs"
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
                >
                  <Plus className="h-4 w-4" />
                  {t('home.new')}
                </Link>
              </div>

              {jobs.length === 0 ? (
                <Card className="p-8 bg-card border-border text-center">
                  <Folder className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-foreground mb-2">
                    {t('home.no_kb_yet')}
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    {t('home.create_first_kb')}
                  </p>
                  <Link
                    to="/jobs"
                    className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
                  >
                    <Plus className="h-4 w-4" />
                    {t('home.create_kb')}
                  </Link>
                </Card>
              ) : (
                <>
                  {/* User Stats */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                    <Card className="p-6 bg-card border-border">
                      <div className="flex items-center gap-3 mb-2">
                        <Folder className="w-5 h-5 text-primary" />
                        <span className="text-muted-foreground text-sm">{t('home.stats_kbs')}</span>
                      </div>
                      <p className="text-3xl font-bold text-white">{jobs.length}</p>
                    </Card>
                    <Card className="p-6 bg-card border-border">
                      <div className="flex items-center gap-3 mb-2">
                        <FileText className="w-5 h-5 text-primary" />
                        <span className="text-muted-foreground text-sm">{t('home.stats_docs')}</span>
                      </div>
                      <p className="text-3xl font-bold text-white">{userTotalDocs}</p>
                    </Card>
                    <Card className="p-6 bg-card border-border">
                      <div className="flex items-center gap-3 mb-2">
                        <Layers className="w-5 h-5 text-primary" />
                        <span className="text-muted-foreground text-sm">{t('home.stats_chunks')}</span>
                      </div>
                      <p className="text-3xl font-bold text-white">{userTotalChunks.toLocaleString()}</p>
                    </Card>
                  </div>

                  {/* Recent Jobs */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {jobs.slice(0, 3).map((job) => (
                      <Link key={job.id} to={`/jobs/${job.id}`}>
                        <Card className="p-5 bg-card border-border hover:border-primary/50 transition-colors cursor-pointer group">
                          <div className="flex items-start gap-3 mb-3">
                            <div className="p-2 bg-primary/10 rounded-lg">
                              <Folder className="h-5 w-5 text-primary" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <h3 className="font-semibold text-foreground truncate">
                                {job.name}
                              </h3>
                              {job.description && (
                                <p className="text-sm text-muted-foreground line-clamp-1 mt-1">
                                  {job.description}
                                </p>
                              )}
                            </div>
                            <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                          </div>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span>{job.document_count} {t('kb.docs')}</span>
                            <span>{job.chunk_count} {t('kb.chunks')}</span>
                          </div>
                        </Card>
                      </Link>
                    ))}
                    {jobs.length > 3 && (
                      <Link to="/jobs">
                        <Card className="p-5 bg-card border-border hover:border-primary/50 transition-colors cursor-pointer flex items-center justify-center h-full min-h-[120px]">
                          <div className="text-center">
                            <p className="text-muted-foreground">
                              {t('home.more', { count: jobs.length - 3 })}
                            </p>
                            <p className="text-sm text-primary mt-1">{t('home.view_all')}</p>
                          </div>
                        </Card>
                      </Link>
                    )}
                  </div>
                </>
              )}
            </motion.section>

            {/* Public Demo Section (collapsed for logged-in users) */}
            <motion.section variants={itemVariants} className="mb-12">
              <div className="border-t border-border pt-8 mt-8">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 rounded-lg bg-secondary/50 flex items-center justify-center">
                    <Database className="w-5 h-5 text-muted-foreground" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-foreground">{t('home.demo_title')}</h2>
                    <p className="text-sm text-muted-foreground">
                      {t('home.demo_subtitle', { count: stats?.total_papers || 0 })}
                    </p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <Link to="/chats">
                    <Card className="px-6 py-3 bg-card border-border hover:border-primary/50 transition-colors cursor-pointer">
                      <div className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
                        <MessageSquare className="w-4 h-4" />
                        <span className="text-sm">{t('home.demo_chat')}</span>
                      </div>
                    </Card>
                  </Link>
                  <Link to="/datasets">
                    <Card className="px-6 py-3 bg-card border-border hover:border-primary/50 transition-colors cursor-pointer">
                      <div className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
                        <Database className="w-4 h-4" />
                        <span className="text-sm">{t('home.demo_browse')}</span>
                      </div>
                    </Card>
                  </Link>
                </div>
              </div>
            </motion.section>
          </>
        ) : (
          <>
            {/* Guest Mode: Show Default Collection Stats */}
            <motion.section variants={itemVariants} className="mb-12">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
              ) : error ? (
                <div className="text-center py-12 text-destructive">
                  {t('home.stats_error')}: {error}
                </div>
              ) : stats ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                  <Card className="p-6 bg-card border-border">
                    <div className="flex items-center gap-3 mb-2">
                      <FileText className="w-5 h-5 text-primary" />
                      <span className="text-muted-foreground text-sm">{t('home.guest_total_papers')}</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{stats.total_papers}</p>
                  </Card>

                  <Card className="p-6 bg-card border-border">
                    <div className="flex items-center gap-3 mb-2">
                      <Layers className="w-5 h-5 text-primary" />
                      <span className="text-muted-foreground text-sm">{t('home.guest_total_chunks')}</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{stats.total_chunks.toLocaleString()}</p>
                  </Card>

                  <Card className="p-6 bg-card border-border">
                    <div className="flex items-center gap-3 mb-2">
                      <Database className="w-5 h-5 text-primary" />
                      <span className="text-muted-foreground text-sm">{t('home.guest_phases')}</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{Object.keys(stats.phases).length}</p>
                  </Card>

                  <Card className="p-6 bg-card border-border">
                    <div className="flex items-center gap-3 mb-2">
                      <Calendar className="w-5 h-5 text-primary" />
                      <span className="text-muted-foreground text-sm">{t('home.guest_year_range')}</span>
                    </div>
                    <p className="text-3xl font-bold text-white">
                      {stats.year_range.min}-{stats.year_range.max}
                    </p>
                  </Card>
                </div>
              ) : null}
            </motion.section>

            {/* Topics & Phases */}
            {stats && (
              <motion.section variants={itemVariants} className="mb-12">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="p-6 bg-card border-border">
                    <h3 className="text-lg font-semibold text-white mb-4">Topics</h3>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(stats.topics).map(([topic, count]) => (
                        <Badge key={topic} variant="secondary" className="bg-secondary/50">
                          {topic} ({count})
                        </Badge>
                      ))}
                    </div>
                  </Card>

                  <Card className="p-6 bg-card border-border">
                    <h3 className="text-lg font-semibold text-white mb-4">Phases</h3>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(stats.phases).map(([phase, count]) => (
                        <Badge key={phase} variant="secondary" className="bg-secondary/50">
                          {phase} ({count})
                        </Badge>
                      ))}
                    </div>
                  </Card>
                </div>
              </motion.section>
            )}

            {/* Call to Action */}
            <motion.section variants={itemVariants} className="mb-12">
              <Card className="p-8 bg-gradient-to-r from-primary/10 to-accent/10 border-primary/20">
                <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                  <div>
                    <h3 className="text-xl font-semibold text-foreground mb-2">
                      Create Your Own Knowledge Base
                    </h3>
                    <p className="text-muted-foreground">
                      Sign in to upload your own documents and build custom RAG applications.
                    </p>
                  </div>
                  <Link
                    to="/login"
                    className="flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors whitespace-nowrap"
                  >
                    Get Started
                    <ChevronRight className="h-4 w-4" />
                  </Link>
                </div>
              </Card>
            </motion.section>

            {/* Dataset Section */}
            <motion.section variants={itemVariants} className="mb-12">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Database className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-semibold text-white">Dataset</h2>
              </div>

              <Link to="/datasets">
                <motion.div
                  whileHover={{ y: -4, borderColor: 'hsl(var(--primary))' }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="w-full max-w-[280px] h-[100px] bg-card border-border hover:border-primary/50 transition-colors cursor-pointer flex items-center justify-center group">
                    <div className="flex items-center gap-2 text-muted-foreground group-hover:text-foreground transition-colors">
                      <span className="text-sm font-medium">See All Papers</span>
                      <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </Card>
                </motion.div>
              </Link>
            </motion.section>

            {/* Chat Apps Section */}
            <motion.section variants={itemVariants}>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                    <MessageSquare className="w-5 h-5 text-primary" />
                  </div>
                  <h2 className="text-2xl font-semibold text-white">Chat Apps</h2>
                </div>

                {/* App Type Tabs */}
                <div className="hidden sm:flex items-center gap-1 bg-secondary/50 rounded-lg p-1">
                  <Link to="/chats">
                    <button className="px-4 py-1.5 rounded-md text-sm font-medium bg-background text-foreground transition-colors">
                      Chat Apps
                    </button>
                  </Link>
                  <Link to="/searches">
                    <button className="px-4 py-1.5 rounded-md text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                      Search Apps
                    </button>
                  </Link>
                  <Link to="/agents">
                    <button className="px-4 py-1.5 rounded-md text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                      Agent
                    </button>
                  </Link>
                </div>
              </div>

              <Link to="/chats">
                <motion.div
                  whileHover={{ y: -4, borderColor: 'hsl(var(--primary))' }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="w-full max-w-[280px] h-[100px] bg-card border-border hover:border-primary/50 transition-colors cursor-pointer flex items-center justify-center group">
                    <div className="flex items-center gap-2 text-muted-foreground group-hover:text-foreground transition-colors">
                      <span className="text-sm font-medium">Start Chat</span>
                      <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </Card>
                </motion.div>
              </Link>
            </motion.section>
          </>
        )}
      </div>
    </motion.div>
  );
}
