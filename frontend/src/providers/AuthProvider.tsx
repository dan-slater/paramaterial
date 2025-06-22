import { useEffect, ReactNode } from 'react';
import { useAuthStore } from '../stores/authStore';

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const initializeAuth = useAuthStore(state => state.initializeAuth);

  useEffect(() => {
    // Initialize authentication state on app startup
    initializeAuth();
  }, [initializeAuth]);

  return <>{children}</>;
};