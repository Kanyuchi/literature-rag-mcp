import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';
import type { UserResponse, TokenResponse } from '../lib/api';

// Storage keys
const ACCESS_TOKEN_KEY = 'retrievo_access_token';
const REFRESH_TOKEN_KEY = 'retrievo_refresh_token';

interface AuthContextType {
  user: UserResponse | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  accessToken: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name?: string) => Promise<void>;
  logout: () => Promise<void>;
  handleOAuthCallback: (provider: 'google' | 'github', code: string, state?: string) => Promise<void>;
  refreshAuth: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Save tokens to localStorage
  const saveTokens = useCallback((tokens: TokenResponse) => {
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
    localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
    setAccessToken(tokens.access_token);
  }, []);

  // Clear tokens from localStorage
  const clearTokens = useCallback(() => {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    setAccessToken(null);
    setUser(null);
  }, []);

  // Fetch current user with token
  const fetchUser = useCallback(async (token: string): Promise<UserResponse | null> => {
    try {
      const userData = await api.getCurrentUser(token);
      return userData;
    } catch (error) {
      console.error('Failed to fetch user:', error);
      return null;
    }
  }, []);

  // Refresh authentication
  const refreshAuth = useCallback(async (): Promise<boolean> => {
    const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
    if (!refreshToken) {
      return false;
    }

    try {
      const tokens = await api.refreshToken(refreshToken);
      saveTokens(tokens);
      const userData = await fetchUser(tokens.access_token);
      if (userData) {
        setUser(userData);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to refresh auth:', error);
      clearTokens();
      return false;
    }
  }, [saveTokens, fetchUser, clearTokens]);

  // Initialize auth state on mount
  useEffect(() => {
    const initAuth = async () => {
      const storedAccessToken = localStorage.getItem(ACCESS_TOKEN_KEY);

      if (storedAccessToken) {
        setAccessToken(storedAccessToken);
        const userData = await fetchUser(storedAccessToken);

        if (userData) {
          setUser(userData);
        } else {
          // Access token expired, try refresh
          const refreshed = await refreshAuth();
          if (!refreshed) {
            clearTokens();
          }
        }
      }

      setIsLoading(false);
    };

    initAuth();
  }, [fetchUser, refreshAuth, clearTokens]);

  // Login with email/password
  const login = useCallback(async (email: string, password: string) => {
    const tokens = await api.login(email, password);
    saveTokens(tokens);
    const userData = await fetchUser(tokens.access_token);
    if (userData) {
      setUser(userData);
    } else {
      throw new Error('Failed to fetch user data after login');
    }
  }, [saveTokens, fetchUser]);

  // Register new user
  const register = useCallback(async (email: string, password: string, name?: string) => {
    const tokens = await api.register(email, password, name);
    saveTokens(tokens);
    const userData = await fetchUser(tokens.access_token);
    if (userData) {
      setUser(userData);
    } else {
      throw new Error('Failed to fetch user data after registration');
    }
  }, [saveTokens, fetchUser]);

  // Logout
  const logout = useCallback(async () => {
    const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
    if (refreshToken) {
      try {
        await api.logout(refreshToken);
      } catch (error) {
        console.error('Logout API call failed:', error);
      }
    }
    clearTokens();
  }, [clearTokens]);

  // Handle OAuth callback
  const handleOAuthCallback = useCallback(async (
    provider: 'google' | 'github',
    code: string,
    state?: string
  ) => {
    const tokens = await api.handleOAuthCallback(provider, code, state);
    saveTokens(tokens);
    const userData = await fetchUser(tokens.access_token);
    if (userData) {
      setUser(userData);
    } else {
      throw new Error('Failed to fetch user data after OAuth login');
    }
  }, [saveTokens, fetchUser]);

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    accessToken,
    login,
    register,
    logout,
    handleOAuthCallback,
    refreshAuth,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
