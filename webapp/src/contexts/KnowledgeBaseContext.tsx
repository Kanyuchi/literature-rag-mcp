import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useAuth } from './AuthContext';
import { api } from '../lib/api';
import type { Job } from '../lib/api';

// Special ID for the default public collection
export const DEFAULT_COLLECTION_ID = 'default';

export interface KnowledgeBase {
  id: string | number;
  name: string;
  description?: string;
  document_count: number;
  chunk_count: number;
  isDefault: boolean;
}

interface KnowledgeBaseContextType {
  // Currently selected knowledge base
  selectedKB: KnowledgeBase | null;
  // All available knowledge bases (user's jobs + default)
  availableKBs: KnowledgeBase[];
  // Loading state
  isLoading: boolean;
  // Select a knowledge base
  selectKB: (id: string | number) => void;
  // Refresh the list
  refreshKBs: () => Promise<void>;
  // Check if using default collection
  isDefaultSelected: boolean;
}

const KnowledgeBaseContext = createContext<KnowledgeBaseContextType | undefined>(undefined);

// Default public collection info
const DEFAULT_KB: KnowledgeBase = {
  id: DEFAULT_COLLECTION_ID,
  name: 'Public Demo Collection',
  description: 'German regional economic transitions (85 papers)',
  document_count: 85,
  chunk_count: 14590,
  isDefault: true,
};

export function KnowledgeBaseProvider({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, accessToken } = useAuth();
  const [selectedKB, setSelectedKB] = useState<KnowledgeBase | null>(DEFAULT_KB);
  const [availableKBs, setAvailableKBs] = useState<KnowledgeBase[]>([DEFAULT_KB]);
  const [isLoading, setIsLoading] = useState(false);

  // Load user's jobs when authenticated
  const refreshKBs = useCallback(async () => {
    setIsLoading(true);
    try {
      // Always include the default collection
      const kbs: KnowledgeBase[] = [DEFAULT_KB];

      // If authenticated, load user's jobs
      if (isAuthenticated && accessToken) {
        const response = await api.listJobs(accessToken);
        const userKBs = response.jobs.map((job: Job) => ({
          id: job.id,
          name: job.name,
          description: job.description || undefined,
          document_count: job.document_count,
          chunk_count: job.chunk_count,
          isDefault: false,
        }));
        kbs.push(...userKBs);
      }

      setAvailableKBs(kbs);

      // If currently selected KB no longer exists, reset to default
      if (selectedKB && !selectedKB.isDefault) {
        const stillExists = kbs.some(kb => kb.id === selectedKB.id);
        if (!stillExists) {
          setSelectedKB(DEFAULT_KB);
        }
      }
    } catch (err) {
      console.error('Failed to load knowledge bases:', err);
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated, accessToken, selectedKB]);

  // Load KBs on auth change
  useEffect(() => {
    refreshKBs();
  }, [isAuthenticated, accessToken]);

  // Reset to default when logging out
  useEffect(() => {
    if (!isAuthenticated) {
      setSelectedKB(DEFAULT_KB);
      setAvailableKBs([DEFAULT_KB]);
    }
  }, [isAuthenticated]);

  const selectKB = useCallback((id: string | number) => {
    const kb = availableKBs.find(k => k.id === id);
    if (kb) {
      setSelectedKB(kb);
      // Persist selection
      if (id === DEFAULT_COLLECTION_ID) {
        localStorage.removeItem('selected_kb');
      } else {
        localStorage.setItem('selected_kb', String(id));
      }
    }
  }, [availableKBs]);

  // Restore selection from localStorage
  useEffect(() => {
    const savedId = localStorage.getItem('selected_kb');
    if (savedId && isAuthenticated) {
      const numId = parseInt(savedId, 10);
      const kb = availableKBs.find(k => k.id === numId);
      if (kb) {
        setSelectedKB(kb);
      }
    }
  }, [availableKBs, isAuthenticated]);

  const value: KnowledgeBaseContextType = {
    selectedKB,
    availableKBs,
    isLoading,
    selectKB,
    refreshKBs,
    isDefaultSelected: selectedKB?.isDefault ?? true,
  };

  return (
    <KnowledgeBaseContext.Provider value={value}>
      {children}
    </KnowledgeBaseContext.Provider>
  );
}

export function useKnowledgeBase() {
  const context = useContext(KnowledgeBaseContext);
  if (context === undefined) {
    throw new Error('useKnowledgeBase must be used within a KnowledgeBaseProvider');
  }
  return context;
}
