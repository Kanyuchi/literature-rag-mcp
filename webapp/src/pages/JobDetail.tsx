import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { api } from '../lib/api';
import type { Job, JobDocument, JobStats, UploadConfigResponse } from '../lib/api';
import {
  ArrowLeft,
  Upload,
  FileText,
  Trash2,
  Search,
  MessageSquare,
  BarChart3,
  X,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Plus,
} from 'lucide-react';

interface UploadQueueItem {
  id: string;
  file: File;
  phase: string;
  topic: string;
  status: 'pending' | 'uploading' | 'completed' | 'failed';
  error?: string;
  result?: {
    doc_id: string;
    chunks_indexed: number;
  };
}

export default function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { isAuthenticated, isLoading: authLoading, accessToken } = useAuth();

  const [job, setJob] = useState<Job | null>(null);
  const [documents, setDocuments] = useState<JobDocument[]>([]);
  const [stats, setStats] = useState<JobStats | null>(null);
  const [uploadConfig, setUploadConfig] = useState<UploadConfigResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Upload state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadQueue, setUploadQueue] = useState<UploadQueueItem[]>([]);
  const [uploadPhase, setUploadPhase] = useState('');
  const [uploadTopic, setUploadTopic] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  // Query state
  const [showQueryPanel, setShowQueryPanel] = useState(false);
  const [queryText, setQueryText] = useState('');
  const [queryResult, setQueryResult] = useState<string | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  // Delete state
  const [deleteDoc, setDeleteDoc] = useState<JobDocument | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Clear knowledge base state
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [isClearing, setIsClearing] = useState(false);

  // Search/filter
  const [searchQuery, setSearchQuery] = useState('');

  const numericJobId = jobId ? parseInt(jobId, 10) : null;

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login?redirect=/jobs');
    }
  }, [authLoading, isAuthenticated, navigate]);

  const loadJob = useCallback(async () => {
    if (!accessToken || !numericJobId) return;

    setIsLoading(true);
    setError(null);

    try {
      const [jobData, docsData, statsData, configData] = await Promise.all([
        api.getJob(numericJobId, accessToken),
        api.getJobDocuments(numericJobId, { limit: 100 }, accessToken),
        api.getJobStats(numericJobId, accessToken),
        api.getUploadConfig(),
      ]);

      setJob(jobData);
      setDocuments(docsData.documents);
      setStats(statsData);
      setUploadConfig(configData);

      if (configData.phases.length > 0 && !uploadPhase) {
        setUploadPhase(configData.phases[0].name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load job');
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, numericJobId, uploadPhase]);

  useEffect(() => {
    if (accessToken && numericJobId) {
      loadJob();
    }
  }, [accessToken, numericJobId, loadJob]);

  // Handle file selection for multi-upload
  const handleFilesSelected = (files: FileList | null) => {
    if (!files) return;

    const newItems: UploadQueueItem[] = Array.from(files).map((file, index) => ({
      id: `${Date.now()}_${index}`,
      file,
      phase: uploadPhase,
      topic: uploadTopic,
      status: 'pending',
    }));

    setUploadQueue(prev => [...prev, ...newItems]);
  };

  const removeFromQueue = (id: string) => {
    setUploadQueue(prev => prev.filter(item => item.id !== id));
  };

  const updateQueueItem = (id: string, updates: Partial<UploadQueueItem>) => {
    setUploadQueue(prev =>
      prev.map(item => (item.id === id ? { ...item, ...updates } : item))
    );
  };

  // Process upload queue
  const handleUpload = async () => {
    if (!accessToken || !numericJobId || uploadQueue.length === 0) return;

    setIsUploading(true);

    // Process each file sequentially
    for (const item of uploadQueue) {
      if (item.status !== 'pending') continue;

      updateQueueItem(item.id, { status: 'uploading' });

      try {
        const formData = new FormData();
        formData.append('file', item.file);
        formData.append('phase', item.phase || uploadPhase);
        formData.append('topic', item.topic || uploadTopic);

        const response = await fetch(
          `${import.meta.env.VITE_API_URL || 'http://localhost:8001'}/api/jobs/${numericJobId}/upload`,
          {
            method: 'POST',
            headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
            body: formData,
          }
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
          throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const result = await response.json();

        updateQueueItem(item.id, {
          status: 'completed',
          result: {
            doc_id: result.doc_id,
            chunks_indexed: result.chunks_indexed,
          },
        });
      } catch (err) {
        updateQueueItem(item.id, {
          status: 'failed',
          error: err instanceof Error ? err.message : 'Upload failed',
        });
      }
    }

    setIsUploading(false);

    // Reload data after all uploads
    await loadJob();
  };

  const closeUploadModal = () => {
    if (!isUploading) {
      setShowUploadModal(false);
      setUploadQueue([]);
      setUploadTopic('');
    }
  };

  const handleQuery = async () => {
    if (!queryText.trim() || !accessToken || !numericJobId) return;

    setIsQuerying(true);
    setQueryResult(null);

    try {
      const response = await api.queryJob(numericJobId, queryText, { n_sources: 5 }, accessToken);

      // Format the results
      if (response.results && response.results.length > 0) {
        const formatted = response.results.map((r: { content: string; metadata: { title?: string; authors?: string }; score: number }, i: number) =>
          `**[${i + 1}]** ${r.metadata.title || 'Untitled'} (${r.metadata.authors || 'Unknown'})\n${r.content.substring(0, 300)}...\n*Score: ${r.score}*`
        ).join('\n\n---\n\n');
        setQueryResult(formatted);
      } else if (response.message) {
        setQueryResult(response.message);
      } else {
        setQueryResult('No results found for your query.');
      }
    } catch (err) {
      setQueryResult(`Error: ${err instanceof Error ? err.message : 'Query failed'}`);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleDeleteDocument = async () => {
    if (!deleteDoc || !accessToken || !numericJobId) return;

    setIsDeleting(true);

    try {
      await api.deleteJobDocument(numericJobId, deleteDoc.doc_id, accessToken);
      setDeleteDoc(null);
      await loadJob();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document');
      setDeleteDoc(null);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClearKnowledgeBase = async () => {
    if (!accessToken || !numericJobId) return;

    setIsClearing(true);

    try {
      await api.clearJob(numericJobId, accessToken);
      setShowClearConfirm(false);
      await loadJob();
      // Show success message briefly
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear knowledge base');
      setShowClearConfirm(false);
    } finally {
      setIsClearing(false);
    }
  };

  const filteredDocuments = documents.filter(
    (doc) =>
      doc.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (doc.title?.toLowerCase().includes(searchQuery.toLowerCase()) ?? false)
  );

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  if (authLoading || (!isAuthenticated && !authLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="min-h-screen bg-background p-8">
        <div className="max-w-xl mx-auto">
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-destructive mb-2">
              Error Loading Job
            </h2>
            <p className="text-destructive/80">{error || 'Job not found'}</p>
            <Link
              to="/jobs"
              className="inline-flex items-center gap-2 mt-4 text-primary hover:text-primary/80"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Jobs
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const completedUploads = uploadQueue.filter(i => i.status === 'completed').length;
  const failedUploads = uploadQueue.filter(i => i.status === 'failed').length;
  const pendingUploads = uploadQueue.filter(i => i.status === 'pending' || i.status === 'uploading').length;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                to="/jobs"
                className="p-2 text-muted-foreground hover:text-foreground rounded-lg hover:bg-secondary"
              >
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div>
                <h1 className="text-xl font-bold text-foreground">{job.name}</h1>
                {job.description && (
                  <p className="text-sm text-muted-foreground">{job.description}</p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowQueryPanel(!showQueryPanel)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  showQueryPanel
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted-foreground hover:bg-secondary'
                }`}
              >
                <MessageSquare className="h-5 w-5" />
                Query
              </button>
              <button
                onClick={() => setShowClearConfirm(true)}
                className="flex items-center gap-2 px-4 py-2 text-destructive border border-destructive/30 rounded-lg hover:bg-destructive/10 transition-colors"
                title="Clear all documents from this knowledge base"
              >
                <Trash2 className="h-5 w-5" />
                Clear
              </button>
              <button
                onClick={() => setShowUploadModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
              >
                <Upload className="h-5 w-5" />
                Upload PDF
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-8">
          {/* Main Content */}
          <div className="flex-1">
            {/* Stats Cards */}
            {stats && (
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-card border border-border rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                      <FileText className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-foreground">
                        {stats.document_count}
                      </p>
                      <p className="text-sm text-muted-foreground">Documents</p>
                    </div>
                  </div>
                </div>
                <div className="bg-card border border-border rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-500/10 rounded-lg">
                      <BarChart3 className="h-5 w-5 text-green-500" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-foreground">
                        {stats.chunk_count}
                      </p>
                      <p className="text-sm text-muted-foreground">Chunks</p>
                    </div>
                  </div>
                </div>
                <div className="bg-card border border-border rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-500/10 rounded-lg">
                      <BarChart3 className="h-5 w-5 text-purple-500" />
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-foreground">
                        {Object.keys(stats.topics).length}
                      </p>
                      <p className="text-sm text-muted-foreground">Topics</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Search */}
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search documents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-border rounded-lg bg-card text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
                />
              </div>
            </div>

            {/* Documents List */}
            {documents.length === 0 ? (
              <div className="text-center py-12 bg-card border border-border rounded-lg">
                <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium text-foreground mb-2">
                  No documents yet
                </h3>
                <p className="text-muted-foreground mb-4">
                  Upload your first PDF to get started.
                </p>
                <button
                  onClick={() => setShowUploadModal(true)}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
                >
                  <Upload className="h-5 w-5" />
                  Upload PDF
                </button>
              </div>
            ) : (
              <div className="bg-card border border-border rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead className="bg-secondary/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Document
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Phase / Topic
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Chunks
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Added
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {filteredDocuments.map((doc) => (
                      <tr key={doc.id} className="hover:bg-secondary/30">
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-3">
                            <FileText className="h-5 w-5 text-muted-foreground" />
                            <div>
                              <p className="font-medium text-foreground truncate max-w-xs">
                                {doc.title || doc.filename}
                              </p>
                              {doc.authors && (
                                <p className="text-sm text-muted-foreground truncate max-w-xs">
                                  {doc.authors}
                                </p>
                              )}
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm text-foreground">
                            {doc.phase}
                          </span>
                          {doc.topic_category && (
                            <span className="text-sm text-muted-foreground">
                              {' / '}
                              {doc.topic_category}
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-foreground">
                          {doc.chunk_count}
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {formatDate(doc.created_at)}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            onClick={() => setDeleteDoc(doc)}
                            className="p-1 text-muted-foreground hover:text-destructive transition-colors"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Query Panel */}
          {showQueryPanel && (
            <div className="w-96 bg-card border border-border rounded-lg p-4 h-fit sticky top-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-foreground">Query Knowledge Base</h3>
                <button
                  onClick={() => setShowQueryPanel(false)}
                  className="text-muted-foreground hover:text-foreground"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="space-y-4">
                <textarea
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  placeholder="Ask a question about your documents..."
                  rows={4}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-primary focus:border-transparent resize-none"
                />

                <button
                  onClick={handleQuery}
                  disabled={!queryText.trim() || isQuerying}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isQuerying ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Querying...
                    </>
                  ) : (
                    <>
                      <MessageSquare className="h-4 w-4" />
                      Ask Question
                    </>
                  )}
                </button>

                {queryResult && (
                  <div className="p-4 bg-secondary/50 rounded-lg max-h-96 overflow-y-auto">
                    <p className="text-sm text-foreground whitespace-pre-wrap">
                      {queryResult}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={closeUploadModal} />
          <div className="relative bg-card border border-border rounded-xl shadow-xl max-w-lg w-full mx-4 p-6 max-h-[80vh] overflow-y-auto">
            <h2 className="text-xl font-semibold text-foreground mb-4">
              Upload PDFs
            </h2>

            <div className="space-y-4">
              {/* Phase Select */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Phase
                </label>
                <select
                  value={uploadPhase}
                  onChange={(e) => setUploadPhase(e.target.value)}
                  disabled={isUploading}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground"
                >
                  {uploadConfig?.phases.map((phase) => (
                    <option key={phase.name} value={phase.name}>
                      {phase.name} - {phase.full_name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Topic Input */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Topic
                </label>
                <input
                  type="text"
                  value={uploadTopic}
                  onChange={(e) => setUploadTopic(e.target.value)}
                  placeholder="e.g., Business Formation"
                  disabled={isUploading}
                  list="existing-topics"
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground placeholder:text-muted-foreground"
                />
                <datalist id="existing-topics">
                  {Object.keys(stats?.topics || {}).map((topic) => (
                    <option key={topic} value={topic} />
                  ))}
                </datalist>
              </div>

              {/* File Input */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  PDF Files
                </label>
                <div className="flex items-center gap-2">
                  <label className="flex-1 cursor-pointer">
                    <div className="flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-border rounded-lg hover:border-primary/50 transition-colors">
                      <Plus className="h-5 w-5 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        {uploadQueue.length === 0 ? 'Select PDF files' : 'Add more files'}
                      </span>
                    </div>
                    <input
                      type="file"
                      accept=".pdf"
                      multiple
                      onChange={(e) => handleFilesSelected(e.target.files)}
                      disabled={isUploading}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>

              {/* Upload Queue */}
              {uploadQueue.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-foreground">
                    Files ({uploadQueue.length})
                  </p>
                  <div className="max-h-48 overflow-y-auto space-y-2">
                    {uploadQueue.map((item) => (
                      <div
                        key={item.id}
                        className={`flex items-center gap-3 p-2 rounded-lg ${
                          item.status === 'completed'
                            ? 'bg-green-500/10'
                            : item.status === 'failed'
                            ? 'bg-destructive/10'
                            : item.status === 'uploading'
                            ? 'bg-primary/10'
                            : 'bg-secondary/50'
                        }`}
                      >
                        {item.status === 'completed' ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
                        ) : item.status === 'failed' ? (
                          <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0" />
                        ) : item.status === 'uploading' ? (
                          <Loader2 className="h-4 w-4 text-primary animate-spin flex-shrink-0" />
                        ) : (
                          <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-foreground truncate">
                            {item.file.name}
                          </p>
                          {item.status === 'failed' && item.error && (
                            <p className="text-xs text-destructive">{item.error}</p>
                          )}
                          {item.status === 'completed' && item.result && (
                            <p className="text-xs text-green-500">
                              {item.result.chunks_indexed} chunks indexed
                            </p>
                          )}
                        </div>
                        {item.status === 'pending' && !isUploading && (
                          <button
                            onClick={() => removeFromQueue(item.id)}
                            className="p-1 text-muted-foreground hover:text-destructive"
                          >
                            <X className="h-4 w-4" />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Progress Summary */}
              {isUploading && (
                <div className="p-3 bg-primary/10 border border-primary/20 rounded-lg">
                  <p className="text-sm text-foreground">
                    Uploading... {completedUploads}/{uploadQueue.length} complete
                    {failedUploads > 0 && `, ${failedUploads} failed`}
                  </p>
                </div>
              )}

              <div className="flex justify-end gap-3 pt-2">
                <button
                  onClick={closeUploadModal}
                  disabled={isUploading}
                  className="px-4 py-2 text-muted-foreground hover:bg-secondary rounded-lg transition-colors disabled:opacity-50"
                >
                  {pendingUploads === 0 && completedUploads > 0 ? 'Done' : 'Cancel'}
                </button>
                {pendingUploads > 0 && (
                  <button
                    onClick={handleUpload}
                    disabled={!uploadPhase || !uploadTopic || uploadQueue.length === 0 || isUploading}
                    className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isUploading ? 'Uploading...' : `Upload ${uploadQueue.length} file${uploadQueue.length > 1 ? 's' : ''}`}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => !isDeleting && setDeleteDoc(null)} />
          <div className="relative bg-card border border-border rounded-xl shadow-xl max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-destructive/10 rounded-full">
                <AlertCircle className="h-6 w-6 text-destructive" />
              </div>
              <h2 className="text-xl font-semibold text-foreground">
                Delete Document
              </h2>
            </div>

            <p className="text-muted-foreground mb-6">
              Are you sure you want to delete <strong className="text-foreground">"{deleteDoc.title || deleteDoc.filename}"</strong>?
              This will remove all {deleteDoc.chunk_count} indexed chunks.
            </p>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteDoc(null)}
                disabled={isDeleting}
                className="px-4 py-2 text-muted-foreground hover:bg-secondary rounded-lg transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteDocument}
                disabled={isDeleting}
                className="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isDeleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Clear Knowledge Base Confirmation Modal */}
      {showClearConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => !isClearing && setShowClearConfirm(false)} />
          <div className="relative bg-card border border-border rounded-xl shadow-xl max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-destructive/10 rounded-full">
                <AlertCircle className="h-6 w-6 text-destructive" />
              </div>
              <h2 className="text-xl font-semibold text-foreground">
                Clear Knowledge Base
              </h2>
            </div>

            <p className="text-muted-foreground mb-4">
              Are you sure you want to clear all documents from this knowledge base?
            </p>
            <p className="text-muted-foreground mb-6">
              This will delete <strong className="text-foreground">{documents.length} documents</strong> and{' '}
              <strong className="text-foreground">{stats?.chunk_count || 0} chunks</strong>.
              The knowledge base itself will be preserved, allowing you to upload new documents.
            </p>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowClearConfirm(false)}
                disabled={isClearing}
                className="px-4 py-2 text-muted-foreground hover:bg-secondary rounded-lg transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleClearKnowledgeBase}
                disabled={isClearing}
                className="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isClearing ? 'Clearing...' : 'Clear All'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
