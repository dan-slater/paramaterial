from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import aiofiles
import logging

from database import get_db
from schemas.job import JobCreate, JobUpdate, JobResponse, JobListResponse
from models import User, Job, JobFile
from api.auth import get_current_user
from config_fastapi import get_settings
from services.materials_processor import process_materials_data

router = APIRouter()

@router.get("/", response_model=JobListResponse)
async def list_jobs(
    page: int = 1,
    per_page: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List user's jobs with pagination"""
    settings = get_settings()
    per_page = min(per_page, settings.items_per_page)
    
    # Count total jobs
    count_result = await db.execute(
        select(func.count(Job.id)).where(Job.user_id == current_user.id)
    )
    total = count_result.scalar()
    
    # Get paginated jobs
    offset = (page - 1) * per_page
    result = await db.execute(
        select(Job)
        .where(Job.user_id == current_user.id)
        .order_by(Job.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    jobs = result.scalars().all()
    
    pages = (total + per_page - 1) // per_page
    
    return JobListResponse(
        jobs=[JobResponse.from_orm(job) for job in jobs],
        total=total,
        page=page,
        per_page=per_page,
        pages=pages
    )

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    settings = get_settings()
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in settings.allowed_extensions

@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    background_tasks: BackgroundTasks,
    info_table: UploadFile = File(...),
    time_series_files: List[UploadFile] = File(...),
    organization_id: Optional[str] = Form(None),
    template_id: Optional[str] = Form(None),
    equipment_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new job with file uploads and start background processing"""
    settings = get_settings()
    
    # Validate files
    if not info_table.filename or not allowed_file(info_table.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid info table file is required (csv, xlsx, xls, txt, json)"
        )
    
    if not time_series_files or not time_series_files[0].filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one time series file is required"
        )
    
    for ts_file in time_series_files:
        if ts_file.filename and not allowed_file(ts_file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {ts_file.filename}"
            )
    
    try:
        # Create job record
        job = Job(
            user_id=current_user.id,
            organization_id=organization_id,
            template_id=template_id,
            equipment_id=equipment_id,
            status="uploading"
        )
        db.add(job)
        await db.flush()  # Get the job ID
        
        # Create job directory
        job_dir = os.path.join(settings.upload_folder, job.id)
        os.makedirs(job_dir, exist_ok=True)
        
        saved_files = []
        
        # Save info table file
        info_filename = secure_filename(info_table.filename)
        info_path = os.path.join(job_dir, info_filename)
        
        async with aiofiles.open(info_path, 'wb') as f:
            content = await info_table.read()
            await f.write(content)
        
        # Create info file record
        info_file_record = JobFile(
            job_id=job.id,
            file_name=info_filename,
            file_type="info_table",
            file_size=len(content),
            storage_path=info_path,
            upload_completed=True,
            mime_type=info_table.content_type
        )
        db.add(info_file_record)
        saved_files.append(info_path)
        
        # Save time series files
        for ts_file in time_series_files:
            if ts_file.filename:
                ts_filename = secure_filename(ts_file.filename)
                ts_path = os.path.join(job_dir, ts_filename)
                
                async with aiofiles.open(ts_path, 'wb') as f:
                    ts_content = await ts_file.read()
                    await f.write(ts_content)
                
                ts_file_record = JobFile(
                    job_id=job.id,
                    file_name=ts_filename,
                    file_type="time_series",
                    file_size=len(ts_content),
                    storage_path=ts_path,
                    upload_completed=True,
                    mime_type=ts_file.content_type
                )
                db.add(ts_file_record)
                saved_files.append(ts_path)
        
        # Update job status
        job.status = "pending"
        await db.commit()
        await db.refresh(job)
        
        # Start background processing
        background_tasks.add_task(
            process_materials_data,
            job.id,
            saved_files,
            str(settings.database_url)
        )
        
        logging.info(f"Job {job.id} created and queued for processing")
        
        return JobResponse.from_orm(job)
        
    except Exception as e:
        await db.rollback()
        logging.error(f"Error creating job: {e}")
        
        # Clean up files if they were created
        if 'job_dir' in locals() and os.path.exists(job_dir):
            import shutil
            shutil.rmtree(job_dir, ignore_errors=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating job. Please try again."
        )

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get job details"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Check access
    if job.user_id != current_user.id:
        # TODO: Check organization access
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this job"
        )
    
    return JobResponse.from_orm(job)

@router.get("/{job_id}/status")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get job status for polling"""
    result = await db.execute(
        select(Job.status, Job.error_message, Job.completed_at)
        .where(Job.id == job_id)
    )
    job_data = result.first()
    
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return {
        "job_id": job_id,
        "status": job_data.status,
        "error_message": job_data.error_message,
        "completed_at": job_data.completed_at.isoformat() if job_data.completed_at else None
    }

@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete job and associated files"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Check ownership
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own jobs"
        )
    
    try:
        # Delete files from filesystem
        settings = get_settings()
        job_dir = os.path.join(settings.upload_folder, job.id)
        if os.path.exists(job_dir):
            import shutil
            shutil.rmtree(job_dir, ignore_errors=True)
        
        # Delete job (cascade will delete job_files)
        await db.delete(job)
        await db.commit()
        
        return {"message": "Job deleted successfully"}
        
    except Exception as e:
        await db.rollback()
        logging.error(f"Error deleting job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting job"
        )

# Export router
jobs_router = router