import axios, { AxiosResponse } from 'axios';
import { 
  User, 
  LoginData, 
  RegisterData, 
  TokenResponse, 
  Job, 
  JobListResponse 
} from '../types';

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auth token management
export const setAuthToken = (token: string) => {
  api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  localStorage.setItem('auth_token', token);
};

export const getAuthToken = (): string | null => {
  return localStorage.getItem('auth_token');
};

export const clearAuthToken = () => {
  delete api.defaults.headers.common['Authorization'];
  localStorage.removeItem('auth_token');
  localStorage.removeItem('refresh_token');
};

// Initialize auth token on app start
const token = getAuthToken();
if (token) {
  setAuthToken(token);
}

// Response interceptor for token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401 && getAuthToken()) {
      clearAuthToken();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: async (data: LoginData): Promise<TokenResponse> => {
    const response: AxiosResponse<TokenResponse> = await api.post('/auth/login', data);
    return response.data;
  },

  register: async (data: RegisterData): Promise<User> => {
    const response: AxiosResponse<User> = await api.post('/auth/register', data);
    return response.data;
  },

  getCurrentUser: async (): Promise<User> => {
    const response: AxiosResponse<User> = await api.get('/auth/me');
    return response.data;
  },

  logout: async (): Promise<void> => {
    await api.post('/auth/logout');
  },
};

// Jobs API
export const jobsAPI = {
  getJobs: async (page = 1, perPage = 20): Promise<JobListResponse> => {
    const response: AxiosResponse<JobListResponse> = await api.get('/jobs', {
      params: { page, per_page: perPage },
    });
    return response.data;
  },

  getJob: async (jobId: string): Promise<Job> => {
    const response: AxiosResponse<Job> = await api.get(`/jobs/${jobId}`);
    return response.data;
  },

  getJobStatus: async (jobId: string): Promise<{ job_id: string; status: string; error_message?: string; completed_at?: string }> => {
    const response = await api.get(`/jobs/${jobId}/status`);
    return response.data;
  },

  createJob: async (formData: FormData): Promise<Job> => {
    const response: AxiosResponse<Job> = await api.post('/jobs', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  deleteJob: async (jobId: string): Promise<void> => {
    await api.delete(`/jobs/${jobId}`);
  },
};

// Health check
export const healthAPI = {
  check: async (): Promise<{ status: string; timestamp: string; version: string }> => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  },
};

export default api;