import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronDown, Database, Folder, Plus, Check, Loader2 } from 'lucide-react';
import { useKnowledgeBase, DEFAULT_COLLECTION_ID } from '@/contexts/KnowledgeBaseContext';
import { useAuth } from '@/contexts/AuthContext';

export default function KnowledgeBaseSelector() {
  const { selectedKB, availableKBs, selectKB, isLoading } = useKnowledgeBase();
  const { isAuthenticated } = useAuth();
  const [isOpen, setIsOpen] = useState(false);

  if (!selectedKB) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-secondary/50 hover:bg-secondary text-sm transition-colors max-w-[200px]"
      >
        {selectedKB.isDefault ? (
          <Database className="h-4 w-4 text-primary flex-shrink-0" />
        ) : (
          <Folder className="h-4 w-4 text-primary flex-shrink-0" />
        )}
        <span className="truncate text-foreground">{selectedKB.name}</span>
        {isLoading ? (
          <Loader2 className="h-3 w-3 animate-spin text-muted-foreground flex-shrink-0" />
        ) : (
          <ChevronDown className={`h-3 w-3 text-muted-foreground flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        )}
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full left-0 mt-1 w-72 bg-card border border-border rounded-lg shadow-lg z-50 overflow-hidden">
            <div className="p-2 border-b border-border">
              <p className="text-xs text-muted-foreground px-2">Select Knowledge Base</p>
            </div>

            <div className="max-h-64 overflow-y-auto p-1">
              {availableKBs.map((kb) => (
                <button
                  key={kb.id}
                  onClick={() => {
                    selectKB(kb.id);
                    setIsOpen(false);
                  }}
                  className={`w-full flex items-start gap-3 px-3 py-2 rounded-md text-left transition-colors ${
                    selectedKB.id === kb.id
                      ? 'bg-primary/10 text-foreground'
                      : 'hover:bg-secondary/50 text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {kb.isDefault ? (
                    <Database className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                  ) : (
                    <Folder className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{kb.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {kb.document_count} docs · {kb.chunk_count.toLocaleString()} chunks
                    </p>
                  </div>
                  {selectedKB.id === kb.id && (
                    <Check className="h-4 w-4 text-primary flex-shrink-0 mt-0.5" />
                  )}
                </button>
              ))}
            </div>

            {isAuthenticated && (
              <div className="p-2 border-t border-border">
                <Link
                  to="/jobs"
                  onClick={() => setIsOpen(false)}
                  className="flex items-center gap-2 px-3 py-2 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors"
                >
                  <Plus className="h-4 w-4" />
                  Create New Knowledge Base
                </Link>
              </div>
            )}

            {!isAuthenticated && (
              <div className="p-2 border-t border-border">
                <Link
                  to="/login"
                  onClick={() => setIsOpen(false)}
                  className="flex items-center gap-2 px-3 py-2 rounded-md text-sm text-primary hover:bg-primary/10 transition-colors"
                >
                  Sign in to create your own knowledge base
                </Link>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
