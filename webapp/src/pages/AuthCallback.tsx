import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function AuthCallback() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { handleOAuthCallback } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const processCallback = async () => {
      const code = searchParams.get('code');
      const state = searchParams.get('state');
      const errorParam = searchParams.get('error');
      const errorDescription = searchParams.get('error_description');

      // Check for OAuth error
      if (errorParam) {
        setError(errorDescription || errorParam);
        return;
      }

      if (!code) {
        setError('No authorization code received');
        return;
      }

      // Determine provider from state or URL path
      const provider = state?.startsWith('google') ? 'google' :
                       state?.startsWith('github') ? 'github' : null;

      if (!provider) {
        setError('Could not determine OAuth provider');
        return;
      }

      try {
        await handleOAuthCallback(provider, code, state || undefined);
        navigate('/jobs');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'OAuth authentication failed');
      }
    };

    processCallback();
  }, [searchParams, handleOAuthCallback, navigate]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="max-w-md w-full space-y-4 p-8">
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-destructive mb-2">
              Authentication Failed
            </h2>
            <p className="text-sm text-destructive/80">{error}</p>
          </div>
          <button
            onClick={() => navigate('/login')}
            className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            Back to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-4">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
        <p className="text-muted-foreground">Completing authentication...</p>
      </div>
    </div>
  );
}
