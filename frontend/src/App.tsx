import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './providers/AuthProvider';
import { useAuthStore } from './stores/authStore';
import { Header } from './components/Header';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { JobDetails } from './pages/JobDetails';
import { Login } from './pages/Login';
import { Register } from './pages/Register';
import { NotificationContainer } from './components/NotificationContainer';

function AppContent() {
  const { isAuthenticated, loading } = useAuthStore();

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <div className="flex flex-col min-h-screen w-full bg-gray-50">
      <Header />
      <NotificationContainer />
      <main className="flex-1">
        <Routes>
          {/* Public routes */}
          <Route 
            path="/login" 
            element={!isAuthenticated ? <Login /> : <Navigate to="/dashboard" />} 
          />
          <Route 
            path="/register" 
            element={!isAuthenticated ? <Register /> : <Navigate to="/dashboard" />} 
          />
          
          {/* Protected routes */}
          <Route 
            path="/dashboard" 
            element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/upload" 
            element={isAuthenticated ? <Upload /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/jobs/:jobId" 
            element={isAuthenticated ? <JobDetails /> : <Navigate to="/login" />} 
          />
          
          {/* Default redirect */}
          <Route 
            path="/" 
            element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} 
          />
          
          {/* Catch all - redirect to dashboard or login */}
          <Route 
            path="*" 
            element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} 
          />
        </Routes>
      </main>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 bg-plasma-gradient rounded-2xl flex items-center justify-center mx-auto mb-4 animate-pulse-plasma">
          <span className="text-white font-bold text-xl">PM</span>
        </div>
        <p className="text-gray-600">Loading ParaMaterial...</p>
      </div>
    </div>
  );
}

export function App() {
  return (
    <AuthProvider>
      <Router>
        <AppContent />
      </Router>
    </AuthProvider>
  );
}