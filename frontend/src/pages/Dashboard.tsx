import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Search, Filter, FileText, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { JobCard } from '../components/JobCard';
import { useAuthStore } from '../stores/authStore';
import { useJobsStore } from '../stores/jobsStore';


export function Dashboard() {
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();
  
  const { user } = useAuthStore();
  const { jobs, pagination, loading, fetchJobs } = useJobsStore();

  useEffect(() => {
    fetchJobs(1, 50);
  }, [fetchJobs]);

  // Calculate stats from jobs
  const stats = {
    totalJobs: pagination.total,
    activeJobs: jobs.filter(job => 
      ['uploading', 'pending', 'processing'].includes(job.status)
    ).length,
    completedJobs: jobs.filter(job => job.status === 'completed').length,
    failedJobs: jobs.filter(job => job.status === 'failed').length
  };

  const filteredJobs = jobs.filter(job =>
    job.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    job.files.some(file => file.file_name.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const recentJobs = filteredJobs.slice(0, 6);

  if (loading) {
    return <DashboardSkeleton />;
  }

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="flex flex-col gap-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Welcome back, {user?.first_name}!
            </h1>
            <p className="text-gray-600 mt-1">
              Manage your materials testing data and analysis jobs
            </p>
          </div>
          <button
            onClick={() => navigate('/upload')}
            className="flex items-center gap-2 px-4 py-2 bg-plasma-500 text-white rounded-lg hover:bg-plasma-600 transition-colors duration-200"
          >
            <Plus size={16} />
            New Analysis
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Total Jobs"
            value={stats.totalJobs.toString()}
            icon={<FileText size={20} />}
            color="bg-blue-500"
          />
          <StatCard
            title="Active Jobs"
            value={stats.activeJobs.toString()}
            icon={<Clock size={20} />}
            color="bg-yellow-500"
          />
          <StatCard
            title="Completed"
            value={stats.completedJobs.toString()}
            icon={<CheckCircle size={20} />}
            color="bg-green-500"
          />
          <StatCard
            title="Failed"
            value={stats.failedJobs.toString()}
            icon={<AlertCircle size={20} />}
            color="bg-red-500"
          />
        </div>

        {/* Search and Filter */}
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 relative">
            <Search
              className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
              size={20}
            />
            <input
              type="text"
              placeholder="Search jobs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-plasma-500 focus:border-transparent transition-all duration-200"
            />
          </div>
          <button className="flex items-center gap-2 px-4 py-2 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors duration-200">
            <Filter size={20} />
            Filters
          </button>
        </div>

        {/* Recent Jobs */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Recent Jobs</h2>
            <a
              href="/jobs"
              className="text-plasma-600 hover:text-plasma-700 text-sm font-medium"
            >
              View all jobs →
            </a>
          </div>

          {recentJobs.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recentJobs.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onClick={() => navigate(`/jobs/${job.id}`)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="bg-gradient-to-r from-plasma-500 to-plasma-600 rounded-lg p-6 text-white">
          <h3 className="text-lg font-semibold mb-2">Quick Start</h3>
          <p className="text-plasma-100 mb-4">
            Ready to analyze your materials testing data? Upload your files and get started.
          </p>
          <div className="flex flex-wrap gap-3">
            <a
              href="/upload"
              className="px-4 py-2 bg-white text-plasma-600 rounded-lg hover:bg-gray-100 transition-colors font-medium"
            >
              Upload Data
            </a>
            <a
              href="/templates"
              className="px-4 py-2 bg-plasma-400 text-white rounded-lg hover:bg-plasma-300 transition-colors font-medium"
            >
              Browse Templates
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: string;
}

function StatCard({ title, value, icon, color }: StatCardProps) {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 transition-all duration-200 hover:shadow-md">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${color} text-white`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
      <FileText size={48} className="mx-auto text-gray-400 mb-4" />
      <h3 className="text-lg font-medium text-gray-900 mb-2">No jobs yet</h3>
      <p className="text-gray-600 mb-6">
        Upload your first materials testing data to get started
      </p>
      <a
        href="/upload"
        className="inline-flex items-center gap-2 px-4 py-2 bg-plasma-500 text-white rounded-lg hover:bg-plasma-600 transition-colors"
      >
        <Plus size={16} />
        Create First Job
      </a>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="container mx-auto px-6 py-8">
      <div className="flex flex-col gap-8">
        {/* Header Skeleton */}
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <div className="h-8 bg-gray-200 rounded w-64 animate-pulse"></div>
            <div className="h-4 bg-gray-200 rounded w-80 animate-pulse"></div>
          </div>
          <div className="h-10 bg-gray-200 rounded w-32 animate-pulse"></div>
        </div>

        {/* Stats Skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 animate-pulse">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-20"></div>
                  <div className="h-8 bg-gray-200 rounded w-12"></div>
                </div>
                <div className="h-12 w-12 bg-gray-200 rounded-lg"></div>
              </div>
            </div>
          ))}
        </div>

        {/* Search Skeleton */}
        <div className="flex gap-4">
          <div className="flex-1 h-10 bg-gray-200 rounded-lg animate-pulse"></div>
          <div className="w-24 h-10 bg-gray-200 rounded-lg animate-pulse"></div>
        </div>

        {/* Jobs Skeleton */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <div className="h-6 bg-gray-200 rounded w-32 animate-pulse"></div>
            <div className="h-4 bg-gray-200 rounded w-24 animate-pulse"></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 animate-pulse">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="h-10 w-10 bg-gray-200 rounded-lg"></div>
                      <div className="space-y-2">
                        <div className="h-4 bg-gray-200 rounded w-24"></div>
                        <div className="h-3 bg-gray-200 rounded w-32"></div>
                      </div>
                    </div>
                    <div className="h-6 bg-gray-200 rounded-full w-20"></div>
                  </div>
                  <div className="space-y-2">
                    <div className="h-3 bg-gray-200 rounded w-full"></div>
                    <div className="h-3 bg-gray-200 rounded w-3/4"></div>
                    <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}