import pandas as pd
import logging
from io import BytesIO
from typing import List, Tuple, Optional
from .supabase_client import supabase_client

logger = logging.getLogger(__name__)

class SupabaseValidation:
    def __init__(self):
        self.client = supabase_client.get_client()
    
    def validate_series_files(self, job_id: str, user_id: str, info_ids: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate time series files in Supabase storage against expected IDs
        
        Args:
            job_id: UUID of the job
            user_id: UUID of the user
            info_ids: List of expected IDs from info table
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Get all job files for this job
            result = self.client.table('job_files').select('file_name, file_type').eq('job_id', job_id).execute()
            
            if not result.data:
                errors.append("No files found for this job")
                return False, errors
            
            # Filter for time series files and extract base names
            time_series_files = [
                file['file_name'] for file in result.data 
                if file['file_type'] == 'time_series'
            ]
            
            found_files = []
            for filename in time_series_files:
                # Extract base name (remove extension)
                base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
                found_files.append(base_name)
            
            # Convert to sets for comparison
            info_ids_set = set(map(str, info_ids))
            found_files_set = set(found_files)
            
            # Check for missing files
            missing_series = info_ids_set - found_files_set
            if missing_series:
                errors.append(f"Missing series file(s) for ID(s): {', '.join(sorted(list(missing_series)))}")
            
            # Check for extra files
            extra_files = found_files_set - info_ids_set
            if extra_files:
                errors.append(f"Found extra series file(s) not listed in info table: {', '.join(sorted(list(extra_files)))}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating series files for job {job_id}: {e}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def extract_ids_from_info_table(self, job_id: str, user_id: str, info_filename: str) -> List[str]:
        """
        Download and read info table from Supabase storage to extract IDs
        
        Args:
            job_id: UUID of the job
            user_id: UUID of the user  
            info_filename: Name of the info table file
            
        Returns:
            List of IDs found in the info table
            
        Raises:
            ValueError: If file cannot be read or processed
        """
        try:
            # Generate storage path
            storage_path = f"{user_id}/{job_id}/{info_filename}"
            
            # Download file from Supabase storage
            result = self.client.storage.from_('job-files').download(storage_path)
            
            if not result:
                raise ValueError(f"Could not download info table file: {info_filename}")
            
            # Read file content into pandas DataFrame
            file_content = BytesIO(result)
            
            if info_filename.endswith('.csv'):
                df = pd.read_csv(file_content)
            elif info_filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_content)
            else:
                raise ValueError(f"Unsupported file type: {info_filename}. Please upload CSV or XLSX.")
            
            if df.empty:
                raise ValueError(f"Info table is empty: {info_filename}")
            
            # Look for ID column
            id_column = None
            if 'ID' in df.columns:
                id_column = 'ID'
            elif 'test_id' in df.columns:
                id_column = 'test_id'
            else:
                raise ValueError("Info table must contain an 'ID' or 'test_id' column.")
            
            logger.debug(f"Found ID column: '{id_column}'")
            
            # Extract IDs as strings
            ids = df[id_column].astype(str).tolist()
            logger.debug(f"Extracted {len(ids)} IDs from info table.")
            
            return ids
            
        except Exception as e:
            logger.error(f"Failed to extract IDs from info table '{info_filename}': {e}")
            raise ValueError(f"Error processing info table: {e}")
    
    def validate_job_files(self, job_id: str, user_id: str) -> Tuple[bool, List[str]]:
        """
        Complete validation workflow for a job
        
        Args:
            job_id: UUID of the job
            user_id: UUID of the user
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            # Get job files
            result = self.client.table('job_files').select('file_name, file_type').eq('job_id', job_id).execute()
            
            if not result.data:
                return False, ["No files found for this job"]
            
            # Find info table file
            info_files = [f for f in result.data if f['file_type'] == 'info_table']
            if not info_files:
                return False, ["No info table file found"]
            
            if len(info_files) > 1:
                return False, ["Multiple info table files found - only one is allowed"]
            
            info_filename = info_files[0]['file_name']
            
            # Extract IDs from info table
            expected_ids = self.extract_ids_from_info_table(job_id, user_id, info_filename)
            
            # Validate time series files
            return self.validate_series_files(job_id, user_id, expected_ids)
            
        except ValueError as e:
            return False, [str(e)]
        except Exception as e:
            logger.error(f"Unexpected error during job validation: {e}")
            return False, [f"Validation failed: {str(e)}"]

# Global instance
validation = SupabaseValidation()