import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
}

interface UIState {
  // State
  sidebarOpen: boolean;
  notifications: Notification[];
  theme: 'light' | 'dark';
  loading: {
    global: boolean;
    [key: string]: boolean;
  };
  
  // Actions
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  setLoading: (key: string, loading: boolean) => void;
  setGlobalLoading: (loading: boolean) => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    (set, get) => ({
      // Initial state
      sidebarOpen: false,
      notifications: [],
      theme: 'light',
      loading: {
        global: false,
      },

      // Actions
      toggleSidebar: () => {
        set(
          (state) => ({ sidebarOpen: !state.sidebarOpen }),
          false,
          'ui/toggleSidebar'
        );
      },

      setSidebarOpen: (open: boolean) => {
        set({ sidebarOpen: open }, false, 'ui/setSidebarOpen');
      },

      addNotification: (notification: Omit<Notification, 'id'>) => {
        const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
        const newNotification: Notification = {
          id,
          duration: 5000, // 5 seconds default
          ...notification,
        };
        
        set(
          (state) => ({
            notifications: [newNotification, ...state.notifications],
          }),
          false,
          'ui/addNotification'
        );

        // Auto-remove notification after duration
        if (newNotification.duration && newNotification.duration > 0) {
          setTimeout(() => {
            get().removeNotification(id);
          }, newNotification.duration);
        }
      },

      removeNotification: (id: string) => {
        set(
          (state) => ({
            notifications: state.notifications.filter(notif => notif.id !== id),
          }),
          false,
          'ui/removeNotification'
        );
      },

      clearNotifications: () => {
        set({ notifications: [] }, false, 'ui/clearNotifications');
      },

      setTheme: (theme: 'light' | 'dark') => {
        set({ theme }, false, 'ui/setTheme');
        
        // Update document class for Tailwind dark mode
        if (theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
        
        // Persist theme preference
        localStorage.setItem('theme', theme);
      },

      setLoading: (key: string, loading: boolean) => {
        set(
          (state) => ({
            loading: {
              ...state.loading,
              [key]: loading,
            },
          }),
          false,
          'ui/setLoading'
        );
      },

      setGlobalLoading: (loading: boolean) => {
        set(
          (state) => ({
            loading: {
              ...state.loading,
              global: loading,
            },
          }),
          false,
          'ui/setGlobalLoading'
        );
      },
    }),
    {
      name: 'ui-store',
    }
  )
);

// Initialize theme from localStorage
const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null;
if (savedTheme) {
  useUIStore.getState().setTheme(savedTheme);
} else {
  // Check system preference
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  useUIStore.getState().setTheme(systemPrefersDark ? 'dark' : 'light');
}