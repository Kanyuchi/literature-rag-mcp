import { useState, useEffect, useCallback, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  FolderOpen, Search, Upload, ChevronLeft, ChevronRight,
  FileText, Trash2, X, AlertCircle, CheckCircle, Loader2
} from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { api, DocumentInfo, UploadConfigResponse } from '@/lib/api';

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

export default function Files() {
  // State
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadConfig, setUploadConfig] = useState<UploadConfigResponse | null>(null);

  // Upload dialog state
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPhase, setSelectedPhase] = useState('');
  const [selectedTopic, setSelectedTopic] = useState('');
  const [customTopic, setCustomTopic] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Drag and drop state
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(50);

  // Delete confirmation state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<DocumentInfo | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Fetch documents and upload config
  useEffect(() => {
    loadDocuments();
    loadUploadConfig();
  }, []);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const response = await api.listDocuments({ limit: 500 });
      setDocuments(response.documents);
    } catch (error) {
      console.error('Failed to load documents:', error);
      toast.error('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const loadUploadConfig = async () => {
    try {
      const config = await api.getUploadConfig();
      setUploadConfig(config);
    } catch (error) {
      console.error('Failed to load upload config:', error);
    }
  };

  // Drag and drop handlers
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const pdfFile = files.find(f => f.name.toLowerCase().endsWith('.pdf'));

    if (pdfFile) {
      setSelectedFile(pdfFile);
      setUploadDialogOpen(true);
    } else {
      toast.error('Please drop a PDF file');
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.name.toLowerCase().endsWith('.pdf')) {
        setSelectedFile(file);
        setUploadDialogOpen(true);
      } else {
        toast.error('Please select a PDF file');
      }
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !selectedPhase) {
      toast.error('Please select a file and phase');
      return;
    }

    const topic = customTopic || selectedTopic;
    if (!topic) {
      toast.error('Please select or enter a topic');
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(0);

      const result = await api.uploadPDF(
        selectedFile,
        selectedPhase,
        topic,
        (progress) => setUploadProgress(progress)
      );

      if (result.success) {
        toast.success(`Successfully indexed ${result.filename}`, {
          description: `${result.chunks_indexed} chunks created`,
        });
        setUploadDialogOpen(false);
        resetUploadForm();
        loadDocuments(); // Refresh the list
      } else {
        toast.error('Upload failed', {
          description: result.error || 'Unknown error',
        });
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Upload failed', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const resetUploadForm = () => {
    setSelectedFile(null);
    setSelectedPhase('');
    setSelectedTopic('');
    setCustomTopic('');
  };

  const handleDeleteClick = (doc: DocumentInfo) => {
    setDocumentToDelete(doc);
    setDeleteDialogOpen(true);
  };

  const handleDelete = async () => {
    if (!documentToDelete) return;

    try {
      setDeleting(true);
      const result = await api.deleteDocument(documentToDelete.doc_id);

      if (result.success) {
        toast.success('Document deleted', {
          description: `Removed ${result.chunks_deleted} chunks`,
        });
        setDeleteDialogOpen(false);
        setDocumentToDelete(null);
        loadDocuments(); // Refresh the list
      } else {
        toast.error('Delete failed', {
          description: result.error || 'Unknown error',
        });
      }
    } catch (error) {
      console.error('Delete error:', error);
      toast.error('Delete failed', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setDeleting(false);
    }
  };

  // Filter documents by search
  const filteredDocuments = documents.filter(doc => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      doc.title?.toLowerCase().includes(query) ||
      doc.authors?.toLowerCase().includes(query) ||
      doc.filename?.toLowerCase().includes(query) ||
      doc.topic_category?.toLowerCase().includes(query)
    );
  });

  // Pagination
  const totalPages = Math.ceil(filteredDocuments.length / itemsPerPage);
  const paginatedDocuments = filteredDocuments.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const formatFileSize = (pages?: number) => {
    if (!pages) return '-';
    return `${pages} pages`;
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-[calc(100vh-72px)] bg-background px-4 md:px-8 lg:px-12 py-6"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="fixed inset-0 z-50 bg-primary/20 backdrop-blur-sm flex items-center justify-center">
          <div className="bg-card border-2 border-dashed border-primary rounded-xl p-12 text-center">
            <Upload className="w-16 h-16 text-primary mx-auto mb-4" />
            <p className="text-xl font-semibold text-foreground">Drop PDF here</p>
            <p className="text-muted-foreground mt-2">to upload and index</p>
          </div>
        </div>
      )}

      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div
          variants={itemVariants}
          className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <FolderOpen className="w-5 h-5 text-primary" />
            </div>
            <h1 className="text-2xl font-semibold text-white">Files</h1>
            <span className="text-sm text-muted-foreground">
              ({documents.length} documents)
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-[200px] bg-secondary/50 border-border focus:border-primary"
              />
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              className="hidden"
            />

            <Button
              className="bg-white text-background hover:bg-white/90 gap-2"
              onClick={() => fileInputRef.current?.click()}
              disabled={!uploadConfig?.enabled}
            >
              <Upload className="w-4 h-4" />
              Add file
            </Button>
          </div>
        </motion.div>

        {/* Files Table */}
        <motion.div variants={itemVariants}>
          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="bg-secondary/30 hover:bg-secondary/30 border-border">
                  <TableHead className="w-12">
                    <Checkbox className="border-border" />
                  </TableHead>
                  <TableHead className="text-muted-foreground font-medium">Name</TableHead>
                  <TableHead className="text-muted-foreground font-medium">Phase</TableHead>
                  <TableHead className="text-muted-foreground font-medium">Topic</TableHead>
                  <TableHead className="text-muted-foreground font-medium">Year</TableHead>
                  <TableHead className="text-muted-foreground font-medium">Size</TableHead>
                  <TableHead className="text-muted-foreground font-medium text-right">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-12">
                      <Loader2 className="w-8 h-8 text-muted-foreground animate-spin mx-auto" />
                      <p className="text-muted-foreground mt-2">Loading documents...</p>
                    </TableCell>
                  </TableRow>
                ) : paginatedDocuments.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-12">
                      <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        {searchQuery ? 'No documents match your search' : 'No documents yet'}
                      </p>
                      {!searchQuery && (
                        <Button
                          variant="outline"
                          className="mt-4"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          <Upload className="w-4 h-4 mr-2" />
                          Upload your first PDF
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedDocuments.map((doc) => (
                    <TableRow
                      key={doc.doc_id}
                      className="border-border hover:bg-secondary/20 transition-colors"
                    >
                      <TableCell>
                        <Checkbox className="border-border" />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-3">
                          <FileText className="w-5 h-5 text-red-400" />
                          <div className="min-w-0">
                            <p className="text-foreground truncate max-w-[300px]" title={doc.title || doc.filename}>
                              {doc.title || doc.filename}
                            </p>
                            {doc.authors && (
                              <p className="text-xs text-muted-foreground truncate max-w-[300px]">
                                {doc.authors}
                              </p>
                            )}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-muted-foreground">{doc.phase || '-'}</TableCell>
                      <TableCell className="text-muted-foreground">{doc.topic_category || '-'}</TableCell>
                      <TableCell className="text-muted-foreground">{doc.year || '-'}</TableCell>
                      <TableCell className="text-muted-foreground">{formatFileSize(doc.total_pages)}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-400 hover:text-red-300 hover:bg-red-400/10"
                          onClick={() => handleDeleteClick(doc)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </motion.div>

        {/* Pagination */}
        <motion.div
          variants={itemVariants}
          className="flex items-center justify-end gap-4 mt-8 pt-4 border-t border-border"
        >
          <span className="text-sm text-muted-foreground">Total {filteredDocuments.length}</span>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              className="border-border bg-secondary/50 hover:bg-secondary"
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" className="border-border bg-secondary/50 hover:bg-secondary">
              {currentPage}
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="border-border bg-secondary/50 hover:bg-secondary"
              disabled={currentPage >= totalPages}
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="border-border bg-secondary/50 hover:bg-secondary gap-2">
                {itemsPerPage} / Page <ChevronLeft className="w-4 h-4 rotate-90" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="bg-card border-border">
              {[10, 20, 50, 100].map(num => (
                <DropdownMenuItem key={num} onClick={() => setItemsPerPage(num)}>
                  {num} / Page
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </motion.div>
      </div>

      {/* Upload Dialog */}
      <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Upload PDF</DialogTitle>
            <DialogDescription>
              Select phase and topic for the document to be indexed properly.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {/* Selected file */}
            {selectedFile && (
              <div className="flex items-center gap-3 p-3 bg-secondary/50 rounded-lg">
                <FileText className="w-8 h-8 text-red-400" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium truncate">{selectedFile.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedFile(null)}
                  disabled={uploading}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            )}

            {/* Phase selection */}
            <div className="space-y-2">
              <Label>Phase *</Label>
              <Select value={selectedPhase} onValueChange={setSelectedPhase} disabled={uploading}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a phase" />
                </SelectTrigger>
                <SelectContent>
                  {uploadConfig?.phases.map(phase => (
                    <SelectItem key={phase.name} value={phase.name}>
                      {phase.name} - {phase.full_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Topic selection */}
            <div className="space-y-2">
              <Label>Topic *</Label>
              <Select value={selectedTopic} onValueChange={setSelectedTopic} disabled={uploading}>
                <SelectTrigger>
                  <SelectValue placeholder="Select existing topic or enter new" />
                </SelectTrigger>
                <SelectContent>
                  {uploadConfig?.existing_topics.map(topic => (
                    <SelectItem key={topic} value={topic}>
                      {topic}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="text-xs text-muted-foreground">or enter a new topic:</div>
              <Input
                placeholder="Enter new topic name..."
                value={customTopic}
                onChange={(e) => setCustomTopic(e.target.value)}
                disabled={uploading}
              />
            </div>

            {/* Upload progress */}
            {uploading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Uploading...</span>
                  <span className="text-foreground">{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-2" />
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setUploadDialogOpen(false);
                resetUploadForm();
              }}
              disabled={uploading}
            >
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={uploading || !selectedFile || !selectedPhase || (!selectedTopic && !customTopic)}
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload & Index
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-red-400">
              <AlertCircle className="w-5 h-5" />
              Delete Document
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this document? This will remove all indexed chunks from the knowledge base.
            </DialogDescription>
          </DialogHeader>

          {documentToDelete && (
            <div className="py-4">
              <div className="flex items-center gap-3 p-3 bg-secondary/50 rounded-lg">
                <FileText className="w-8 h-8 text-red-400" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium truncate">
                    {documentToDelete.title || documentToDelete.filename}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {documentToDelete.phase} - {documentToDelete.topic_category}
                  </p>
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setDeleteDialogOpen(false);
                setDocumentToDelete(null);
              }}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleting}
            >
              {deleting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
