import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { Job, JobListResponse, JobStatus } from '../types';
import { jobsAPI } from '../utils/api';

interface JobsState {
  // State
  jobs: Job[];
  currentJob: Job | null;
  pagination: {
    total: number;
    page: number;
    per_page: number;
    pages: number;
  };
  loading: boolean;
  uploading: boolean;
  error: string | null;
  
  // Actions
  fetchJobs: (page?: number, perPage?: number) => Promise<void>;
  fetchJob: (jobId: string) => Promise<void>;
  createJob: (formData: FormData) => Promise<Job>;
  deleteJob: (jobId: string) => Promise<void>;
  pollJobStatus: (jobId: string) => Promise<void>;
  updateJobStatus: (jobId: string, status: JobStatus, error?: string) => void;
  clearCurrentJob: () => void;
  clearError: () => void;
}

export const useJobsStore = create<JobsState>()(
  devtools(
    (set, get) => ({
      // Initial state
      jobs: [],
      currentJob: null,
      pagination: {
        total: 0,
        page: 1,
        per_page: 20,
        pages: 0,
      },
      loading: false,
      uploading: false,
      error: null,

      // Actions
      fetchJobs: async (page = 1, perPage = 20) => {
        set({ loading: true, error: null }, false, 'jobs/fetchJobs/start');
        
        try {
          const response: JobListResponse = await jobsAPI.getJobs(page, perPage);
          
          set({ 
            jobs: response.jobs,
            pagination: {
              total: response.total,
              page: response.page,
              per_page: response.per_page,
              pages: response.pages,
            },
            loading: false,
            error: null
          }, false, 'jobs/fetchJobs/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to fetch jobs';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'jobs/fetchJobs/error');
        }
      },

      fetchJob: async (jobId: string) => {
        set({ loading: true, error: null }, false, 'jobs/fetchJob/start');
        
        try {
          const job = await jobsAPI.getJob(jobId);
          
          set({ 
            currentJob: job,
            loading: false,
            error: null
          }, false, 'jobs/fetchJob/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to fetch job';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'jobs/fetchJob/error');
        }
      },

      createJob: async (formData: FormData) => {
        set({ uploading: true, error: null }, false, 'jobs/createJob/start');
        
        try {
          const job = await jobsAPI.createJob(formData);
          
          // Add new job to the beginning of the list
          const { jobs } = get();
          set({ 
            jobs: [job, ...jobs],
            currentJob: job,
            uploading: false,
            error: null
          }, false, 'jobs/createJob/success');
          
          return job;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to create job';
          set({ 
            uploading: false, 
            error: errorMessage 
          }, false, 'jobs/createJob/error');
          throw error;
        }
      },

      deleteJob: async (jobId: string) => {
        set({ loading: true, error: null }, false, 'jobs/deleteJob/start');
        
        try {
          await jobsAPI.deleteJob(jobId);
          
          // Remove job from the list
          const { jobs } = get();
          const updatedJobs = jobs.filter(job => job.id !== jobId);
          
          set({ 
            jobs: updatedJobs,
            currentJob: get().currentJob?.id === jobId ? null : get().currentJob,
            loading: false,
            error: null
          }, false, 'jobs/deleteJob/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to delete job';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'jobs/deleteJob/error');
        }
      },

      pollJobStatus: async (jobId: string) => {
        try {
          const statusResponse = await jobsAPI.getJobStatus(jobId);
          
          // Update job status in the list
          const { jobs, currentJob } = get();
          const updatedJobs = jobs.map(job => 
            job.id === jobId 
              ? { 
                  ...job, 
                  status: statusResponse.status as JobStatus,
                  error_message: statusResponse.error_message,
                  completed_at: statusResponse.completed_at
                }
              : job
          );
          
          const updatedCurrentJob = currentJob?.id === jobId
            ? { 
                ...currentJob, 
                status: statusResponse.status as JobStatus,
                error_message: statusResponse.error_message,
                completed_at: statusResponse.completed_at
              }
            : currentJob;
          
          set({ 
            jobs: updatedJobs,
            currentJob: updatedCurrentJob
          }, false, 'jobs/pollJobStatus');
          
        } catch (error) {
          console.error('Failed to poll job status:', error);
        }
      },

      updateJobStatus: (jobId: string, status: JobStatus, error?: string) => {
        const { jobs, currentJob } = get();
        
        const updatedJobs = jobs.map(job => 
          job.id === jobId 
            ? { 
                ...job, 
                status,
                error_message: error,
                completed_at: status === 'completed' ? new Date().toISOString() : job.completed_at
              }
            : job
        );
        
        const updatedCurrentJob = currentJob?.id === jobId
          ? { 
              ...currentJob, 
              status,
              error_message: error,
              completed_at: status === 'completed' ? new Date().toISOString() : currentJob.completed_at
            }
          : currentJob;
        
        set({ 
          jobs: updatedJobs,
          currentJob: updatedCurrentJob
        }, false, 'jobs/updateJobStatus');
      },

      clearCurrentJob: () => {
        set({ currentJob: null }, false, 'jobs/clearCurrentJob');
      },

      clearError: () => {
        set({ error: null }, false, 'jobs/clearError');
      },
    }),
    {
      name: 'jobs-store',
    }
  )
);