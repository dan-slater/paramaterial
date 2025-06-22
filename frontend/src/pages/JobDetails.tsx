import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  Download, 
  RefreshCw, 
  Clock, 
  CheckCircle, 
  AlertCircle, 
  FileText,
  BarChart3,
  Trash2
} from 'lucide-react';
import { Job } from '../types';
import { jobsAPI } from '../utils/api';

export function JobDetails() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (jobId) {
      loadJob();
      
      // Start polling if job is in progress
      const pollInterval = setInterval(() => {
        if (job?.status && ['uploading', 'pending', 'processing'].includes(job.status)) {
          pollJobStatus();
        }
      }, 3000); // Poll every 3 seconds

      return () => clearInterval(pollInterval);
    }
  }, [jobId, job?.status]);

  const loadJob = async () => {
    if (!jobId) return;
    
    try {
      setLoading(true);
      const jobData = await jobsAPI.getJob(jobId);
      setJob(jobData);
    } catch (error: any) {
      console.error('Failed to load job:', error);
      setError(error.response?.data?.detail || 'Failed to load job details');
    } finally {
      setLoading(false);
    }
  };

  const pollJobStatus = async () => {
    if (!jobId) return;
    
    try {
      setPolling(true);
      const statusData = await jobsAPI.getJobStatus(jobId);
      
      if (job && statusData.status !== job.status) {
        // Status changed, reload full job data
        await loadJob();
      }
    } catch (error) {
      console.error('Failed to poll job status:', error);
    } finally {
      setPolling(false);
    }
  };

  const handleDelete = async () => {
    if (!jobId || !job) return;
    
    if (window.confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
      try {
        await jobsAPI.deleteJob(jobId);
        navigate('/dashboard');
      } catch (error: any) {
        console.error('Failed to delete job:', error);
        setError(error.response?.data?.detail || 'Failed to delete job');
      }
    }
  };

  if (loading) {
    return <JobDetailsSkeleton />;
  }

  if (error || !job) {
    return (
      <div className="container mx-auto px-6 py-8">
        <div className="text-center py-12">
          <AlertCircle size={48} className="mx-auto text-red-500 mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            {error || 'Job not found'}
          </h2>
          <a
            href="/dashboard"
            className="text-plasma-600 hover:text-plasma-700 font-medium"
          >
            ← Back to Dashboard
          </a>
        </div>
      </div>
    );
  }

  const statusConfig = {
    uploading: { icon: <Clock size={20} />, color: 'text-blue-600', bg: 'bg-blue-100' },
    pending: { icon: <Clock size={20} />, color: 'text-yellow-600', bg: 'bg-yellow-100' },
    processing: { icon: <RefreshCw size={20} className="animate-spin" />, color: 'text-blue-600', bg: 'bg-blue-100' },
    completed: { icon: <CheckCircle size={20} />, color: 'text-green-600', bg: 'bg-green-100' },
    failed: { icon: <AlertCircle size={20} />, color: 'text-red-600', bg: 'bg-red-100' },
    cancelled: { icon: <AlertCircle size={20} />, color: 'text-gray-600', bg: 'bg-gray-100' },
  };

  const status = statusConfig[job.status as keyof typeof statusConfig];

  return (
    <div className="container mx-auto px-6 py-8 max-w-6xl">
      <div className="flex flex-col gap-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <a
              href="/dashboard"
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft size={20} />
            </a>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Job {job.id.slice(0, 8)}...
              </h1>
              <p className="text-gray-600 mt-1">
                Created {new Date(job.created_at).toLocaleString()}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {polling && (
              <RefreshCw size={16} className="animate-spin text-gray-400" />
            )}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${status.bg}`}>
              <span className={status.color}>{status.icon}</span>
              <span className={`font-medium ${status.color} capitalize`}>
                {job.status}
              </span>
            </div>
            
            {job.status === 'completed' && (
              <button className="flex items-center gap-2 px-4 py-2 bg-plasma-500 text-white rounded-lg hover:bg-plasma-600 transition-colors">
                <Download size={16} />
                Export Results
              </button>
            )}
            
            <button
              onClick={handleDelete}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
            >
              <Trash2 size={16} />
              Delete
            </button>
          </div>
        </div>

        {/* Status Message */}
        {job.error_message && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <AlertCircle size={20} className="text-red-500 mt-0.5" />
              <div>
                <h3 className="font-medium text-red-900">Processing Error</h3>
                <p className="text-red-700 mt-1">{job.error_message}</p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Files Section */}
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <FileText size={20} />
                Uploaded Files
              </h2>
              <div className="space-y-3">
                {job.files.map((file) => (
                  <div
                    key={file.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <FileText size={16} className="text-gray-500" />
                      <div>
                        <p className="font-medium text-gray-900">{file.file_name}</p>
                        <p className="text-sm text-gray-500">
                          {file.file_type === 'info_table' ? 'Info Table' : 'Time Series'} • 
                          {Math.round(file.file_size / 1024)} KB
                        </p>
                      </div>
                    </div>
                    <CheckCircle size={16} className="text-green-500" />
                  </div>
                ))}
              </div>
            </div>

            {/* Results Section */}
            {job.status === 'completed' && job.results && (
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <BarChart3 size={20} />
                  Analysis Results
                </h2>
                
                {/* Summary Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-sm font-medium text-blue-900">Data Points</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {job.results.data_points?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <p className="text-sm font-medium text-green-900">Processing Time</p>
                    <p className="text-2xl font-bold text-green-600">
                      {job.results.processing_time ? `${job.results.processing_time.toFixed(2)}s` : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <p className="text-sm font-medium text-purple-900">Files Processed</p>
                    <p className="text-2xl font-bold text-purple-600">
                      {job.results.summary_statistics?.total_files || job.files.length}
                    </p>
                  </div>
                  <div className="bg-orange-50 p-4 rounded-lg">
                    <p className="text-sm font-medium text-orange-900">Columns</p>
                    <p className="text-2xl font-bold text-orange-600">
                      {job.results.summary_statistics?.column_count || 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Detailed Results */}
                {job.results.time_series_data && (
                  <div>
                    <h3 className="font-medium text-gray-900 mb-3">Time Series Analysis</h3>
                    <div className="space-y-4">
                      {job.results.time_series_data.map((ts: any, index: number) => (
                        <div key={index} className="bg-gray-50 p-4 rounded-lg">
                          <h4 className="font-medium text-gray-900 mb-2">{ts.filename}</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <span className="text-gray-600">Shape:</span>
                              <span className="ml-2 font-medium">
                                {ts.shape ? `${ts.shape[0]} × ${ts.shape[1]}` : 'N/A'}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-600">Data Points:</span>
                              <span className="ml-2 font-medium">{ts.data_points || 'N/A'}</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Columns:</span>
                              <span className="ml-2 font-medium">{ts.columns?.length || 'N/A'}</span>
                            </div>
                            {ts.time_range && (
                              <div>
                                <span className="text-gray-600">Duration:</span>
                                <span className="ml-2 font-medium">{ts.time_range.duration}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Job Info */}
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h3 className="font-semibold text-gray-900 mb-4">Job Information</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <span className="text-gray-600">Job ID:</span>
                  <span className="ml-2 font-mono text-xs">{job.id}</span>
                </div>
                <div>
                  <span className="text-gray-600">Created:</span>
                  <span className="ml-2">{new Date(job.created_at).toLocaleString()}</span>
                </div>
                {job.completed_at && (
                  <div>
                    <span className="text-gray-600">Completed:</span>
                    <span className="ml-2">{new Date(job.completed_at).toLocaleString()}</span>
                  </div>
                )}
                <div>
                  <span className="text-gray-600">Files:</span>
                  <span className="ml-2">{job.files.length} files</span>
                </div>
              </div>
            </div>

            {/* Configuration */}
            {(job.organization_id || job.template_id || job.equipment_id) && (
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 className="font-semibold text-gray-900 mb-4">Configuration</h3>
                <div className="space-y-3 text-sm">
                  {job.organization_id && (
                    <div>
                      <span className="text-gray-600">Organization:</span>
                      <span className="ml-2">{job.organization_id}</span>
                    </div>
                  )}
                  {job.equipment_id && (
                    <div>
                      <span className="text-gray-600">Equipment:</span>
                      <span className="ml-2">{job.equipment_id}</span>
                    </div>
                  )}
                  {job.template_id && (
                    <div>
                      <span className="text-gray-600">Template:</span>
                      <span className="ml-2">{job.template_id}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function JobDetailsSkeleton() {
  return (
    <div className="container mx-auto px-6 py-8 max-w-6xl">
      <div className="flex flex-col gap-8">
        {/* Header Skeleton */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 bg-gray-200 rounded-lg animate-pulse"></div>
            <div className="space-y-2">
              <div className="h-8 bg-gray-200 rounded w-48 animate-pulse"></div>
              <div className="h-4 bg-gray-200 rounded w-64 animate-pulse"></div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-10 bg-gray-200 rounded-lg w-24 animate-pulse"></div>
            <div className="h-10 bg-gray-200 rounded-lg w-32 animate-pulse"></div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content Skeleton */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 animate-pulse">
              <div className="h-6 bg-gray-200 rounded w-32 mb-4"></div>
              <div className="space-y-3">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-16 bg-gray-100 rounded-lg"></div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar Skeleton */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 animate-pulse">
              <div className="h-6 bg-gray-200 rounded w-24 mb-4"></div>
              <div className="space-y-3">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="h-4 bg-gray-200 rounded"></div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}