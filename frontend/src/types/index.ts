// API Types
export interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  is_verified: boolean;
  created_at: string;
  updated_at?: string;
}

export interface LoginData {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export type JobStatus = 
  | "uploading" 
  | "pending" 
  | "processing" 
  | "completed" 
  | "failed" 
  | "cancelled";

export interface JobFile {
  id: string;
  job_id: string;
  file_name: string;
  file_type: "info_table" | "time_series";
  file_size: number;
  storage_path: string;
  upload_completed: boolean;
  uploaded_at: string;
  mime_type?: string;
}

export interface Job {
  id: string;
  user_id: string;
  organization_id?: string;
  template_id?: string;
  equipment_id?: string;
  status: JobStatus;
  created_at: string;
  updated_at?: string;
  started_at?: string;
  completed_at?: string;
  analysis_config?: Record<string, any>;
  results?: Record<string, any>;
  error_message?: string;
  files: JobFile[];
}

export interface JobListResponse {
  jobs: Job[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

export interface Organization {
  id: string;
  name: string;
  description?: string;
  website?: string;
  location?: string;
  created_at: string;
  updated_at?: string;
  member_count: number;
}

export interface Equipment {
  id: string;
  name: string;
  model?: string;
  manufacturer?: string;
  description?: string;
  capabilities?: Record<string, any>;
}

export interface AnalysisTemplate {
  id: string;
  name: string;
  description?: string;
  equipment_id?: string;
  template_data: Record<string, any>;
  is_public: boolean;
}