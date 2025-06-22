import React from 'react';
import { Clock, FileText, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import { Job, JobStatus } from '../types';

interface JobCardProps {
  job: Job;
  onClick?: () => void;
}

const statusConfig: Record<JobStatus, { 
  icon: React.ReactNode; 
  color: string; 
  bgColor: string;
  label: string;
}> = {
  uploading: {
    icon: <Loader size={16} className="animate-spin" />,
    color: 'text-blue-600',
    bgColor: 'bg-blue-100',
    label: 'Uploading'
  },
  pending: {
    icon: <Clock size={16} />,
    color: 'text-yellow-600',
    bgColor: 'bg-yellow-100',
    label: 'Pending'
  },
  processing: {
    icon: <Loader size={16} className="animate-spin" />,
    color: 'text-blue-600',
    bgColor: 'bg-blue-100',
    label: 'Processing'
  },
  completed: {
    icon: <CheckCircle size={16} />,
    color: 'text-green-600',
    bgColor: 'bg-green-100',
    label: 'Completed'
  },
  failed: {
    icon: <AlertCircle size={16} />,
    color: 'text-red-600',
    bgColor: 'bg-red-100',
    label: 'Failed'
  },
  cancelled: {
    icon: <AlertCircle size={16} />,
    color: 'text-gray-600',
    bgColor: 'bg-gray-100',
    label: 'Cancelled'
  },
};

export function JobCard({ job, onClick }: JobCardProps) {
  const status = statusConfig[job.status];
  const createdDate = new Date(job.created_at).toLocaleDateString();
  const createdTime = new Date(job.created_at).toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });

  const fileCount = job.files.length;
  const infoFile = job.files.find(f => f.file_type === 'info_table');
  const timeSeriesFiles = job.files.filter(f => f.file_type === 'time_series');

  return (
    <div 
      className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-all duration-200 cursor-pointer"
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-plasma-gradient rounded-lg flex items-center justify-center">
            <FileText size={20} className="text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">
              Job {job.id.slice(0, 8)}...
            </h3>
            <p className="text-sm text-gray-500">
              {createdDate} at {createdTime}
            </p>
          </div>
        </div>

        <div className={`flex items-center space-x-1 px-3 py-1 rounded-full ${status.bgColor}`}>
          <span className={status.color}>
            {status.icon}
          </span>
          <span className={`text-sm font-medium ${status.color}`}>
            {status.label}
          </span>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Files:</span>
          <span className="font-medium">{fileCount} files</span>
        </div>

        {infoFile && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Info table:</span>
            <span className="font-medium truncate max-w-32" title={infoFile.file_name}>
              {infoFile.file_name}
            </span>
          </div>
        )}

        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Time series:</span>
          <span className="font-medium">{timeSeriesFiles.length} files</span>
        </div>

        {job.error_message && (
          <div className="mt-3 p-2 bg-red-50 rounded text-sm text-red-600">
            {job.error_message}
          </div>
        )}

        {job.results && (
          <div className="mt-3 flex items-center justify-between text-sm">
            <span className="text-gray-600">Data points:</span>
            <span className="font-medium text-green-600">
              {job.results.data_points?.toLocaleString() || 'N/A'}
            </span>
          </div>
        )}
      </div>

      {job.status === 'completed' && job.completed_at && (
        <div className="mt-4 pt-3 border-t border-gray-100">
          <p className="text-xs text-gray-500">
            Completed {new Date(job.completed_at).toLocaleString()}
          </p>
        </div>
      )}
    </div>
  );
}