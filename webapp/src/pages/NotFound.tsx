import { Link } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function NotFound() {
  return (
    <div className="min-h-[calc(100vh-72px)] bg-background px-4 py-10">
      <div className="mx-auto max-w-2xl rounded-2xl border border-border bg-card p-8 text-center">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-destructive/10">
          <AlertTriangle className="h-6 w-6 text-destructive" />
        </div>
        <h1 className="mb-2 text-2xl font-semibold text-foreground">Page not found</h1>
        <p className="mb-6 text-sm text-muted-foreground">
          The page you requested does not exist or may have moved.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Button asChild variant="default">
            <Link to="/">
              <Home className="mr-2 h-4 w-4" />
              Go Home
            </Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/jobs">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Open Knowledge Bases
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
