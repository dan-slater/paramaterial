from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from models import db, Job, JobFile, Organization, Equipment, AnalysisTemplate
from models.activity import ActivityLog
import os
import uuid
from datetime import datetime

jobs_bp = Blueprint('jobs', __name__)

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@jobs_bp.route('/')
@login_required
def list_jobs():
    """List user's jobs"""
    page = request.args.get('page', 1, type=int)
    per_page = current_app.config.get('ITEMS_PER_PAGE', 20)
    
    jobs = Job.query.filter_by(user_id=current_user.id)\
                   .order_by(Job.created_at.desc())\
                   .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('jobs/list.html', jobs=jobs)

@jobs_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create_job():
    """Create new job with file upload"""
    if request.method == 'POST':
        # Get organization context
        organization_id = request.form.get('organization_id')
        template_id = request.form.get('template_id')
        equipment_id = request.form.get('equipment_id')
        
        # Validate files
        if 'info_table' not in request.files:
            flash('Info Table file is required.', 'error')
            return render_template('jobs/create.html')
        
        info_file = request.files['info_table']
        if info_file.filename == '' or not allowed_file(info_file.filename):
            flash('Valid Info Table file is required.', 'error')
            return render_template('jobs/create.html')
        
        series_files = request.files.getlist('time_series_files[]')
        if not series_files or series_files[0].filename == '':
            flash('At least one Time Series file is required.', 'error')
            return render_template('jobs/create.html')
        
        # Validate all time series files
        for ts_file in series_files:
            if ts_file.filename != '' and not allowed_file(ts_file.filename):
                flash(f'Invalid file type: {ts_file.filename}', 'error')
                return render_template('jobs/create.html')
        
        try:
            # Create job
            job = Job(
                user_id=current_user.id,
                organization_id=organization_id if organization_id else None,
                template_id=template_id if template_id else None,
                equipment_id=equipment_id if equipment_id else None,
                status='uploading'
            )
            db.session.add(job)
            db.session.flush()  # Get the job ID
            
            # Create job directory
            job_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], job.id)
            os.makedirs(job_dir, exist_ok=True)
            
            # Save info table file
            info_filename = secure_filename(info_file.filename)
            info_path = os.path.join(job_dir, info_filename)
            info_file.save(info_path)
            
            # Create job file record
            info_file_record = JobFile(
                job_id=job.id,
                file_name=info_filename,
                file_type='info_table',
                file_size=os.path.getsize(info_path),
                storage_path=info_path,
                upload_completed=True,
                mime_type=info_file.content_type
            )
            db.session.add(info_file_record)
            
            # Save time series files
            saved_files = []
            for ts_file in series_files:
                if ts_file and ts_file.filename != '':
                    ts_filename = secure_filename(ts_file.filename)
                    ts_path = os.path.join(job_dir, ts_filename)
                    ts_file.save(ts_path)
                    
                    ts_file_record = JobFile(
                        job_id=job.id,
                        file_name=ts_filename,
                        file_type='time_series',
                        file_size=os.path.getsize(ts_path),
                        storage_path=ts_path,
                        upload_completed=True,
                        mime_type=ts_file.content_type
                    )
                    db.session.add(ts_file_record)
                    saved_files.append(ts_filename)
            
            # Update job status
            job.status = 'pending'
            
            # Log activity
            ActivityLog.log_activity(
                user_id=current_user.id,
                organization_id=organization_id,
                action_type='job_created',
                resource_type='job',
                resource_id=job.id,
                details={
                    'info_table': info_filename,
                    'time_series_files': saved_files,
                    'file_count': len(saved_files) + 1
                }
            )
            
            db.session.commit()
            
            flash('Job created successfully! Files uploaded.', 'success')
            return redirect(url_for('jobs.view_job', job_id=job.id))
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error creating job: {e}")
            flash('Error creating job. Please try again.', 'error')
            
            # Clean up files if they were created
            if 'job_dir' in locals() and os.path.exists(job_dir):
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
    
    # Get user's organizations for form
    organizations = current_user.get_organizations()
    
    return render_template('jobs/create.html', organizations=organizations)

@jobs_bp.route('/<job_id>')
@login_required
def view_job(job_id):
    """View job details"""
    job = Job.query.get_or_404(job_id)
    
    # Check access
    if job.user_id != current_user.id:
        # Check if user has access through organization
        if not job.organization_id or not current_user.is_member_of(job.organization_id):
            flash('You do not have access to this job.', 'error')
            return redirect(url_for('jobs.list_jobs'))
    
    return render_template('jobs/view.html', job=job)

@jobs_bp.route('/<job_id>/configure')
@login_required
def configure_job(job_id):
    """Configure job analysis parameters"""
    job = Job.query.get_or_404(job_id)
    
    # Check access and ownership
    if job.user_id != current_user.id:
        flash('You can only configure your own jobs.', 'error')
        return redirect(url_for('jobs.view_job', job_id=job_id))
    
    if job.status not in ['pending', 'failed']:
        flash('This job cannot be configured in its current state.', 'error')
        return redirect(url_for('jobs.view_job', job_id=job_id))
    
    # Get available templates if in organization
    templates = []
    if job.organization_id:
        templates = AnalysisTemplate.query.filter_by(
            organization_id=job.organization_id,
            is_public=True
        ).all()
        
        if job.equipment_id:
            templates = [t for t in templates if t.equipment_id == job.equipment_id]
    
    return render_template('jobs/configure.html', job=job, templates=templates)

@jobs_bp.route('/<job_id>/delete', methods=['POST'])
@login_required
def delete_job(job_id):
    """Delete job and associated files"""
    job = Job.query.get_or_404(job_id)
    
    # Check ownership
    if job.user_id != current_user.id:
        flash('You can only delete your own jobs.', 'error')
        return redirect(url_for('jobs.view_job', job_id=job_id))
    
    try:
        # Delete files from filesystem
        job_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], job.id)
        if os.path.exists(job_dir):
            import shutil
            shutil.rmtree(job_dir, ignore_errors=True)
        
        # Delete job (cascade will delete job_files)
        db.session.delete(job)
        db.session.commit()
        
        flash('Job deleted successfully.', 'success')
        return redirect(url_for('jobs.list_jobs'))
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deleting job {job_id}: {e}")
        flash('Error deleting job. Please try again.', 'error')
        return redirect(url_for('jobs.view_job', job_id=job_id))