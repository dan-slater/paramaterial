import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
import os
import traceback

from models import Job, JobFile

logger = logging.getLogger(__name__)

def process_materials_data(job_id: str, file_paths: List[str], database_url: str):
    """
    Background task to process materials testing data
    
    Args:
        job_id: ID of the job to process
        file_paths: List of file paths to process
        database_url: Database connection string
    """
    # Create synchronous database connection for background task
    engine = create_engine(database_url.replace('postgresql+asyncpg://', 'postgresql://'))
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        try:
            logger.info(f"Starting processing for job {job_id}")
            
            # Update job status to processing
            session.execute(
                update(Job).where(Job.id == job_id).values(status="processing")
            )
            session.commit()
            
            # Process the files
            results = _process_files(file_paths)
            
            # Update job with results
            session.execute(
                update(Job).where(Job.id == job_id).values(
                    status="completed",
                    completed_at=datetime.utcnow(),
                    results=results,
                    metadata={
                        "processed_files": len(file_paths),
                        "processing_time": results.get("processing_time", 0),
                        "data_points": results.get("data_points", 0)
                    }
                )
            )
            session.commit()
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Update job status to failed
            session.execute(
                update(Job).where(Job.id == job_id).values(
                    status="failed",
                    completed_at=datetime.utcnow(),
                    error_message=str(e)
                )
            )
            session.commit()

def _process_files(file_paths: List[str]) -> Dict[str, Any]:
    """
    Process materials testing files and extract data
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        Dictionary containing processing results
    """
    start_time = datetime.utcnow()
    
    try:
        # Separate info table and time series files
        info_file = None
        time_series_files = []
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            if 'info' in filename.lower() or 'table' in filename.lower():
                info_file = file_path
            else:
                time_series_files.append(file_path)
        
        results = {
            "info_data": None,
            "time_series_data": [],
            "summary_statistics": {},
            "data_points": 0,
            "processing_time": 0
        }
        
        # Process info table
        if info_file:
            logger.info(f"Processing info table: {info_file}")
            info_data = _process_info_table(info_file)
            results["info_data"] = info_data
        
        # Process time series files
        total_data_points = 0
        for ts_file in time_series_files:
            logger.info(f"Processing time series file: {ts_file}")
            ts_data = _process_time_series_file(ts_file)
            results["time_series_data"].append(ts_data)
            total_data_points += ts_data.get("data_points", 0)
        
        results["data_points"] = total_data_points
        
        # Calculate summary statistics
        if results["time_series_data"]:
            results["summary_statistics"] = _calculate_summary_stats(results["time_series_data"])
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        results["processing_time"] = processing_time
        
        logger.info(f"Processing completed. Data points: {total_data_points}, Time: {processing_time:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in _process_files: {e}")
        raise

def _process_info_table(file_path: str) -> Dict[str, Any]:
    """Process info table file"""
    try:
        # Determine file type and read accordingly
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            # Try CSV as fallback
            df = pd.read_csv(file_path)
        
        # Convert to dictionary format
        info_data = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "data": df.to_dict('records')[:100],  # Limit to first 100 rows
            "sample_info": {
                "total_samples": len(df),
                "parameters": df.columns.tolist()
            }
        }
        
        return info_data
        
    except Exception as e:
        logger.error(f"Error processing info table {file_path}: {e}")
        return {"error": str(e)}

def _process_time_series_file(file_path: str) -> Dict[str, Any]:
    """Process time series data file"""
    try:
        # Determine file type and read accordingly
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            # Try CSV as fallback
            df = pd.read_csv(file_path)
        
        # Basic time series analysis
        filename = os.path.basename(file_path)
        
        # Assume first column is time, rest are data columns
        data_columns = df.columns[1:] if len(df.columns) > 1 else df.columns
        
        # Calculate basic statistics for each data column
        column_stats = {}
        for col in data_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "count": int(df[col].count())
                }
        
        ts_data = {
            "filename": filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data_points": len(df),
            "column_statistics": column_stats,
            "time_range": {
                "start": df.iloc[0, 0] if len(df) > 0 else None,
                "end": df.iloc[-1, 0] if len(df) > 0 else None,
                "duration": len(df)
            },
            # Store sample of data (first 10 and last 10 rows)
            "sample_data": {
                "head": df.head(10).to_dict('records'),
                "tail": df.tail(10).to_dict('records')
            }
        }
        
        return ts_data
        
    except Exception as e:
        logger.error(f"Error processing time series file {file_path}: {e}")
        return {"filename": os.path.basename(file_path), "error": str(e)}

def _calculate_summary_stats(time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall summary statistics"""
    try:
        total_data_points = sum(ts.get("data_points", 0) for ts in time_series_data)
        total_files = len(time_series_data)
        
        # Collect all numeric columns across files
        all_columns = set()
        for ts in time_series_data:
            if "column_statistics" in ts:
                all_columns.update(ts["column_statistics"].keys())
        
        summary = {
            "total_files": total_files,
            "total_data_points": total_data_points,
            "average_points_per_file": total_data_points / total_files if total_files > 0 else 0,
            "unique_columns": list(all_columns),
            "column_count": len(all_columns),
            "files_processed": [ts.get("filename", "unknown") for ts in time_series_data]
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error calculating summary stats: {e}")
        return {"error": str(e)}