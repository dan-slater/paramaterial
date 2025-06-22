import React, { useCallback, useState } from 'react';
import { Upload, X, FileText, AlertCircle } from 'lucide-react';

interface FileUploadProps {
  onInfoFileChange: (file: File | null) => void;
  onTimeSeriesFilesChange: (files: File[]) => void;
  infoFile: File | null;
  timeSeriesFiles: File[];
}

const ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls', 'txt', 'json'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export function FileUpload({ 
  onInfoFileChange, 
  onTimeSeriesFilesChange, 
  infoFile, 
  timeSeriesFiles 
}: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  const validateFile = (file: File): string | null => {
    const extension = file.name.split('.').pop()?.toLowerCase();
    
    if (!extension || !ALLOWED_EXTENSIONS.includes(extension)) {
      return `Invalid file type: ${file.name}. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`;
    }
    
    if (file.size > MAX_FILE_SIZE) {
      return `File too large: ${file.name}. Max size: 50MB`;
    }
    
    return null;
  };

  const handleFiles = useCallback((files: FileList) => {
    const newErrors: string[] = [];
    const validFiles: File[] = [];

    Array.from(files).forEach(file => {
      const error = validateFile(file);
      if (error) {
        newErrors.push(error);
      } else {
        validFiles.push(file);
      }
    });

    setErrors(newErrors);

    if (validFiles.length > 0) {
      // Add to time series files
      onTimeSeriesFilesChange([...timeSeriesFiles, ...validFiles]);
    }
  }, [timeSeriesFiles, onTimeSeriesFilesChange]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles]);

  const handleInfoFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const error = validateFile(file);
      if (error) {
        setErrors([error]);
      } else {
        setErrors([]);
        onInfoFileChange(file);
      }
    }
  };

  const handleTimeSeriesSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const removeInfoFile = () => {
    onInfoFileChange(null);
  };

  const removeTimeSeriesFile = (index: number) => {
    const newFiles = timeSeriesFiles.filter((_, i) => i !== index);
    onTimeSeriesFilesChange(newFiles);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Info Table Upload */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Info Table (Required) *
        </label>
        <div className="space-y-2">
          <input
            type="file"
            id="info-file"
            accept=".csv,.xlsx,.xls,.txt,.json"
            onChange={handleInfoFileSelect}
            className="hidden"
          />
          <label
            htmlFor="info-file"
            className="flex items-center justify-center px-6 py-4 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-plasma-400 hover:bg-plasma-50 transition-colors"
          >
            <div className="text-center">
              <Upload className="mx-auto h-8 w-8 text-gray-400" />
              <p className="mt-2 text-sm text-gray-600">
                Click to upload info table file
              </p>
              <p className="text-xs text-gray-400">
                CSV, Excel, TXT, or JSON (max 50MB)
              </p>
            </div>
          </label>

          {infoFile && (
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2">
                <FileText size={16} className="text-gray-500" />
                <span className="text-sm font-medium">{infoFile.name}</span>
                <span className="text-xs text-gray-500">
                  ({formatFileSize(infoFile.size)})
                </span>
              </div>
              <button
                onClick={removeInfoFile}
                className="p-1 hover:bg-gray-200 rounded"
              >
                <X size={16} className="text-gray-500" />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Time Series Upload */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Time Series Files (Required) *
        </label>
        <div className="space-y-2">
          <input
            type="file"
            id="time-series-files"
            accept=".csv,.xlsx,.xls,.txt,.json"
            multiple
            onChange={handleTimeSeriesSelect}
            className="hidden"
          />
          <div
            className={`flex items-center justify-center px-6 py-4 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
              dragActive
                ? 'border-plasma-400 bg-plasma-50'
                : 'border-gray-300 hover:border-plasma-400 hover:bg-plasma-50'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('time-series-files')?.click()}
          >
            <div className="text-center">
              <Upload className="mx-auto h-8 w-8 text-gray-400" />
              <p className="mt-2 text-sm text-gray-600">
                Drop files here or click to upload
              </p>
              <p className="text-xs text-gray-400">
                Multiple files supported (CSV, Excel, TXT, JSON)
              </p>
            </div>
          </div>

          {/* Time Series Files List */}
          {timeSeriesFiles.length > 0 && (
            <div className="space-y-2">
              {timeSeriesFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <FileText size={16} className="text-gray-500" />
                    <span className="text-sm font-medium">{file.name}</span>
                    <span className="text-xs text-gray-500">
                      ({formatFileSize(file.size)})
                    </span>
                  </div>
                  <button
                    onClick={() => removeTimeSeriesFile(index)}
                    className="p-1 hover:bg-gray-200 rounded"
                  >
                    <X size={16} className="text-gray-500" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="space-y-2">
          {errors.map((error, index) => (
            <div key={index} className="flex items-center space-x-2 p-3 bg-red-50 rounded-lg">
              <AlertCircle size={16} className="text-red-500 flex-shrink-0" />
              <span className="text-sm text-red-600">{error}</span>
            </div>
          ))}
        </div>
      )}

      {/* Upload Summary */}
      <div className="p-4 bg-blue-50 rounded-lg">
        <h4 className="text-sm font-medium text-blue-900 mb-2">Upload Summary</h4>
        <div className="text-sm text-blue-700 space-y-1">
          <p>Info table: {infoFile ? '✓ Ready' : '⚠ Required'}</p>
          <p>Time series: {timeSeriesFiles.length} files selected</p>
          <p>
            Total size: {formatFileSize(
              (infoFile?.size || 0) + timeSeriesFiles.reduce((sum, f) => sum + f.size, 0)
            )}
          </p>
        </div>
      </div>
    </div>
  );
}