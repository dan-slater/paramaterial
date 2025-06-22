import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Send, Loader } from 'lucide-react';
import { FileUpload } from '../components/FileUpload';
import { useJobsStore } from '../stores/jobsStore';
import { useUIStore } from '../stores/uiStore';
import { Job } from '../types';

export function Upload() {
  const [infoFile, setInfoFile] = useState<File | null>(null);
  const [timeSeriesFiles, setTimeSeriesFiles] = useState<File[]>([]);
  const [organizationId, setOrganizationId] = useState<string>('');
  const [templateId, setTemplateId] = useState<string>('');
  const [equipmentId, setEquipmentId] = useState<string>('');
  
  const navigate = useNavigate();
  const { createJob, uploading, error, clearError } = useJobsStore();
  const { addNotification } = useUIStore();

  const canSubmit = infoFile && timeSeriesFiles.length > 0 && !uploading;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!canSubmit) return;

    clearError();

    try {
      const formData = new FormData();
      
      // Add files
      formData.append('info_table', infoFile!);
      timeSeriesFiles.forEach(file => {
        formData.append('time_series_files', file);
      });

      // Add optional fields
      if (organizationId) formData.append('organization_id', organizationId);
      if (templateId) formData.append('template_id', templateId);
      if (equipmentId) formData.append('equipment_id', equipmentId);

      const job: Job = await createJob(formData);
      
      addNotification({
        type: 'success',
        title: 'Upload successful!',
        message: 'Your data has been uploaded and analysis has started.',
      });
      
      // Redirect to job details page
      navigate(`/jobs/${job.id}`);
      
    } catch (error: any) {
      console.error('Upload failed:', error);
      addNotification({
        type: 'error',
        title: 'Upload failed',
        message: error.response?.data?.detail || 'Please check your files and try again.',
      });
    }
  };

  return (
    <div className="container mx-auto px-6 py-8 max-w-4xl">
      <div className="flex flex-col gap-8">
        {/* Header */}
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/dashboard')}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Upload Analysis Data</h1>
            <p className="text-gray-600 mt-1">
              Upload your materials testing data for analysis and parameterization
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* File Upload Section */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Data Files
            </h2>
            <FileUpload
              infoFile={infoFile}
              timeSeriesFiles={timeSeriesFiles}
              onInfoFileChange={setInfoFile}
              onTimeSeriesFilesChange={setTimeSeriesFiles}
            />
          </div>

          {/* Analysis Configuration */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Analysis Configuration
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Organization (Optional)
                </label>
                <select
                  value={organizationId}
                  onChange={(e) => setOrganizationId(e.target.value)}
                  className="w-full p-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-plasma-500 focus:border-transparent"
                >
                  <option value="">Select organization...</option>
                  <option value="uct-materials">UCT Centre for Materials Engineering</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Equipment (Optional)
                </label>
                <select
                  value={equipmentId}
                  onChange={(e) => setEquipmentId(e.target.value)}
                  className="w-full p-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-plasma-500 focus:border-transparent"
                >
                  <option value="">Select equipment...</option>
                  <option value="tensile-tester-1">Tensile Testing Machine #1</option>
                  <option value="compression-tester">Compression Testing Machine</option>
                  <option value="fatigue-tester">Fatigue Testing Machine</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Template (Optional)
                </label>
                <select
                  value={templateId}
                  onChange={(e) => setTemplateId(e.target.value)}
                  className="w-full p-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-plasma-500 focus:border-transparent"
                >
                  <option value="">Select template...</option>
                  <option value="steel-tensile">Steel Tensile Test</option>
                  <option value="polymer-compression">Polymer Compression</option>
                  <option value="fatigue-analysis">Fatigue Analysis</option>
                </select>
              </div>
            </div>
          </div>

          {/* Analysis Preview */}
          <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
            <h3 className="text-lg font-semibold text-blue-900 mb-4">
              What happens next?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                  1
                </div>
                <div>
                  <p className="font-medium text-blue-900">File Validation</p>
                  <p className="text-blue-700">Check file formats and data integrity</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                  2
                </div>
                <div>
                  <p className="font-medium text-blue-900">Data Processing</p>
                  <p className="text-blue-700">Extract and analyze materials data</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                  3
                </div>
                <div>
                  <p className="font-medium text-blue-900">Results Ready</p>
                  <p className="text-blue-700">View statistics and download results</p>
                </div>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">
                    Upload Error
                  </h3>
                  <div className="mt-2 text-sm text-red-700">
                    {error}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <div className="flex items-center justify-between pt-6 border-t border-gray-200">
            <div className="text-sm text-gray-600">
              {infoFile && timeSeriesFiles.length > 0 ? (
                <>
                  Ready to upload {timeSeriesFiles.length + 1} files
                  {/* Calculate total size */}
                  {' '}({Math.round(((infoFile?.size || 0) + timeSeriesFiles.reduce((sum, f) => sum + f.size, 0)) / 1024 / 1024 * 100) / 100} MB)
                </>
              ) : (
                'Please select files to continue'
              )}
            </div>
            
            <button
              type="submit"
              disabled={!canSubmit}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                canSubmit
                  ? 'bg-plasma-500 text-white hover:bg-plasma-600 shadow-sm hover:shadow-md'
                  : 'bg-gray-200 text-gray-500 cursor-not-allowed'
              }`}
            >
              {uploading ? (
                <>
                  <Loader size={16} className="animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Send size={16} />
                  Start Analysis
                </>
              )}
            </button>
          </div>
        </form>

        {/* Help Section */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Need Help?
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
            <div>
              <p className="font-medium text-gray-900 mb-2">Supported File Formats:</p>
              <ul className="space-y-1">
                <li>• CSV files (.csv)</li>
                <li>• Excel files (.xlsx, .xls)</li>
                <li>• Text files (.txt)</li>
                <li>• JSON files (.json)</li>
              </ul>
            </div>
            <div>
              <p className="font-medium text-gray-900 mb-2">File Requirements:</p>
              <ul className="space-y-1">
                <li>• Maximum file size: 50 MB</li>
                <li>• Info table: Required (1 file)</li>
                <li>• Time series: Required (1+ files)</li>
                <li>• Structured data with headers</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}