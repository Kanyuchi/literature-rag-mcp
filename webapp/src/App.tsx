import { BrowserRouter as Router, Routes, Route, useLocation, Navigate } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { Toaster } from 'sonner';
import { AuthProvider } from './contexts/AuthContext';
import { KnowledgeBaseProvider } from './contexts/KnowledgeBaseContext';
import MainNav from './components/MainNav';
import Home from './pages/Home';
import Dataset from './pages/Dataset';
import Chat from './pages/Chat';
import Search from './pages/Search';
import Agent from './pages/Agent';
import Files from './pages/Files';
import Login from './pages/Login';
import AuthCallback from './pages/AuthCallback';
import Jobs from './pages/Jobs';
import JobDetail from './pages/JobDetail';
import KnowledgeInsights from './pages/KnowledgeInsights';
import KnowledgeGraph from './pages/KnowledgeGraph';
import DataSources from './pages/settings/DataSources';
import ModelProviders from './pages/settings/ModelProviders';
import MCP from './pages/settings/MCP';
import Team from './pages/settings/Team';
import Profile from './pages/settings/Profile';
import NotFound from './pages/NotFound';

// Layout component that conditionally shows MainNav
function AppLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const hideNavRoutes = ['/login', '/auth/callback'];
  const showNav = !hideNavRoutes.some(route => location.pathname.startsWith(route));

  return (
    <div className="min-h-screen bg-background">
      {showNav && <MainNav />}
      <Toaster
        position="top-right"
        theme="dark"
        toastOptions={{
          style: {
            background: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            color: 'hsl(var(--foreground))',
          },
        }}
      />
      <main className={showNav ? "pt-[72px]" : ""}>
        {children}
      </main>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <KnowledgeBaseProvider>
          <AppLayout>
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/datasets" element={<Dataset />} />
              <Route path="/chats" element={<Chat />} />
              <Route path="/searches" element={<Search />} />
              <Route path="/agents" element={<Agent />} />
              <Route path="/files" element={<Files />} />
              <Route path="/login" element={<Login />} />
              <Route path="/auth/callback" element={<AuthCallback />} />
              <Route path="/jobs" element={<Jobs />} />
              <Route path="/jobs/:jobId" element={<JobDetail />} />
              <Route path="/insights" element={<KnowledgeInsights />} />
              <Route path="/graph" element={<KnowledgeGraph />} />
              <Route path="/settings/data-sources" element={<DataSources />} />
              <Route path="/settings/model-providers" element={<ModelProviders />} />
              <Route path="/settings/mcp" element={<MCP />} />
              <Route path="/settings/team" element={<Team />} />
              <Route path="/settings/profile" element={<Profile />} />
              {/* Legacy route aliases */}
              <Route path="/settings/knowledge-insights" element={<Navigate to="/insights" replace />} />
              <Route path="/settings/knowledge-graph" element={<Navigate to="/graph" replace />} />
              <Route path="/settings/files" element={<Navigate to="/files" replace />} />
              <Route path="/docs" element={<Navigate to="/api/docs" replace />} />
              {/* Catch-all fallback */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </AnimatePresence>
        </AppLayout>
        </KnowledgeBaseProvider>
      </AuthProvider>
    </Router>
  );
}

export default App;
