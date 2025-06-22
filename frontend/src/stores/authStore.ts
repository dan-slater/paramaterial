import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { User, LoginData, RegisterData } from '../types';
import { authAPI, setAuthToken, clearAuthToken, getAuthToken } from '../utils/api';

interface AuthState {
  // State
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  
  // Actions
  login: (data: LoginData) => Promise<void>;
  register: (data: RegisterData) => Promise<User | void>;
  logout: () => Promise<void>;
  getCurrentUser: () => Promise<void>;
  initializeAuth: () => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        user: null,
        isAuthenticated: false,
        loading: false,
        error: null,

        // Actions
        login: async (data: LoginData) => {
          set({ loading: true, error: null }, false, 'auth/login/start');
          
          try {
            const tokenResponse = await authAPI.login(data);
            setAuthToken(tokenResponse.access_token);
            
            // Store refresh token
            localStorage.setItem('refresh_token', tokenResponse.refresh_token);
            
            // Get user data
            const userData = await authAPI.getCurrentUser();
            
            set({ 
              user: userData,
              isAuthenticated: true,
              loading: false,
              error: null
            }, false, 'auth/login/success');
            
          } catch (error: any) {
            const errorMessage = error.response?.data?.detail || 'Login failed';
            set({ 
              loading: false, 
              error: errorMessage 
            }, false, 'auth/login/error');
            throw error;
          }
        },

        register: async (data: RegisterData) => {
          set({ loading: true, error: null }, false, 'auth/register/start');
          
          try {
            const userData = await authAPI.register(data);
            set({ 
              loading: false,
              error: null
            }, false, 'auth/register/success');
            return userData;
          } catch (error: any) {
            const errorMessage = error.response?.data?.detail || 'Registration failed';
            set({ 
              loading: false, 
              error: errorMessage 
            }, false, 'auth/register/error');
            throw error;
          }
        },

        logout: async () => {
          set({ loading: true }, false, 'auth/logout/start');
          
          try {
            await authAPI.logout();
          } catch (error) {
            console.error('Logout error:', error);
          } finally {
            clearAuthToken();
            set({ 
              user: null,
              isAuthenticated: false,
              loading: false,
              error: null
            }, false, 'auth/logout/complete');
          }
        },

        getCurrentUser: async () => {
          const token = getAuthToken();
          if (!token) return;

          set({ loading: true }, false, 'auth/getCurrentUser/start');
          
          try {
            const userData = await authAPI.getCurrentUser();
            set({ 
              user: userData,
              isAuthenticated: true,
              loading: false
            }, false, 'auth/getCurrentUser/success');
          } catch (error) {
            console.error('Failed to get current user:', error);
            clearAuthToken();
            set({ 
              user: null,
              isAuthenticated: false,
              loading: false
            }, false, 'auth/getCurrentUser/error');
          }
        },

        initializeAuth: async () => {
          const token = getAuthToken();
          
          if (token) {
            await get().getCurrentUser();
          } else {
            set({ loading: false }, false, 'auth/initialize/no-token');
          }
        },

        clearError: () => {
          set({ error: null }, false, 'auth/clearError');
        },
      }),
      {
        name: 'auth-storage',
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    {
      name: 'auth-store',
    }
  )
);