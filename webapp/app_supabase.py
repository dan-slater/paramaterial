from flask import Flask, render_template, url_for, request, redirect, flash, session, jsonify
import datetime
import os
import uuid
import logging
import traceback
from werkzeug.utils import secure_filename
from functools import wraps

from config import Config
from utils.supabase_client import supabase_client
from utils.validation_supabase import validation

# --- App Configuration ---
app = Flask(__name__)
app.config.from_object(Config)

# Setup enhanced logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if a file has a valid extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def login_required(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current user from session"""
    return session.get('user')

@app.route('/')
def index():
    """Serves the main dashboard page."""
    current_year = datetime.datetime.now().year
    user = get_current_user()
    
    if not user:
        return render_template('dashboard.html', year=current_year, user=None)
    
    # Get user's recent jobs
    try:
        client = supabase_client.get_client()
        result = client.table('jobs').select('*').eq('user_id', user['id']).order('created_at', desc=True).limit(10).execute()
        recent_jobs = result.data if result.data else []
    except Exception as e:
        logger.error(f"Error fetching recent jobs: {e}")
        recent_jobs = []
    
    return render_template('dashboard.html', year=current_year, user=user, recent_jobs=recent_jobs)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    current_year = datetime.datetime.now().year
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('login.html', year=current_year)
        
        try:
            client = supabase_client.get_client()
            auth_response = client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if auth_response.user:
                session['user'] = {
                    'id': auth_response.user.id,
                    'email': auth_response.user.email
                }
                session['access_token'] = auth_response.session.access_token
                flash('Successfully logged in!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid email or password.', 'error')
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Login failed. Please try again.', 'error')
    
    return render_template('login.html', year=current_year)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    current_year = datetime.datetime.now().year
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('register.html', year=current_year)
        
        try:
            client = supabase_client.get_client()
            auth_response = client.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if auth_response.user:
                flash('Registration successful! Please check your email to confirm your account.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Registration failed. Please try again.', 'error')
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('register.html', year=current_year)

@app.route('/logout')
def logout():
    """Handle user logout"""
    try:
        if 'access_token' in session:
            client = supabase_client.get_client()
            client.auth.sign_out()
    except Exception as e:
        logger.error(f"Logout error: {e}")
    
    session.clear()
    flash('Successfully logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    """Handles data upload (Info Table and Time Series files)."""
    current_year = datetime.datetime.now().year
    user = get_current_user()
    
    if request.method == 'POST':
        try:
            logger.debug(f"Processing POST request to /upload for user {user['id']}")
            
            # --- File Handling & Initial Validation ---
            if 'info_table' not in request.files:
                flash('"Info Table" file part missing in the request.', 'error')
                return render_template('upload.html', year=current_year)
                
            info_file = request.files['info_table']
            if info_file.filename == '':
                flash('No Info Table file selected.', 'error')
                return render_template('upload.html', year=current_year)

            # Check Time Series Files
            series_files = request.files.getlist('time_series_files[]')
            if not series_files or (series_files[0].filename == ''):
                flash('No Time Series files selected or provided.', 'error')
                return render_template('upload.html', year=current_year)

            # --- Create Job Record ---
            client = supabase_client.get_client()
            job_data = {
                'user_id': user['id'],
                'status': 'uploading',
                'metadata': {}
            }
            
            job_result = client.table('jobs').insert(job_data).execute()
            if not job_result.data:
                flash('Failed to create job record.', 'error')
                return render_template('upload.html', year=current_year)
            
            job_id = job_result.data[0]['id']
            logger.debug(f"Created job {job_id} for user {user['id']}")

            # --- Upload Files to Supabase Storage ---
            try:
                # Upload Info Table
                info_filename = secure_filename(info_file.filename)
                info_storage_path = f"{user['id']}/{job_id}/{info_filename}"
                
                storage_result = client.storage.from_('job-files').upload(
                    info_storage_path,
                    info_file.read(),
                    {
                        'content-type': info_file.content_type,
                        'upsert': False
                    }
                )
                
                if storage_result.get('error'):
                    raise Exception(f"Failed to upload info table: {storage_result['error']}")
                
                # Record info table file
                file_data = {
                    'job_id': job_id,
                    'file_name': info_filename,
                    'file_type': 'info_table',
                    'file_size': len(info_file.read()),
                    'storage_path': info_storage_path,
                    'upload_completed': True
                }
                client.table('job_files').insert(file_data).execute()
                
                # Upload Time Series Files
                for ts_file in series_files:
                    if ts_file and ts_file.filename != '':
                        ts_filename = secure_filename(ts_file.filename)
                        ts_storage_path = f"{user['id']}/{job_id}/{ts_filename}"
                        
                        ts_content = ts_file.read()
                        storage_result = client.storage.from_('job-files').upload(
                            ts_storage_path,
                            ts_content,
                            {
                                'content-type': ts_file.content_type,
                                'upsert': False
                            }
                        )
                        
                        if storage_result.get('error'):
                            raise Exception(f"Failed to upload {ts_filename}: {storage_result['error']}")
                        
                        # Record time series file
                        file_data = {
                            'job_id': job_id,
                            'file_name': ts_filename,
                            'file_type': 'time_series',
                            'file_size': len(ts_content),
                            'storage_path': ts_storage_path,
                            'upload_completed': True
                        }
                        client.table('job_files').insert(file_data).execute()
                
                logger.debug(f"Uploaded all files for job {job_id}")
                
            except Exception as e:
                logger.error(f"Error uploading files for job {job_id}: {e}")
                # Update job status to failed
                client.table('jobs').update({
                    'status': 'failed',
                    'error_message': f'Upload failed: {str(e)}'
                }).eq('id', job_id).execute()
                flash(f'Error uploading files: {str(e)}', 'error')
                return render_template('upload.html', year=current_year)

            # --- Validation Logic ---
            try:
                # Update job status to validating
                client.table('jobs').update({'status': 'validating'}).eq('id', job_id).execute()
                
                # Validate files
                is_valid, validation_errors = validation.validate_job_files(job_id, user['id'])
                
                if not is_valid:
                    error_message = "Validation Failed: " + "; ".join(validation_errors)
                    logger.warning(f"Validation failed for job {job_id}: {error_message}")
                    
                    # Update job status
                    client.table('jobs').update({
                        'status': 'failed',
                        'error_message': error_message
                    }).eq('id', job_id).execute()
                    
                    flash(error_message, 'danger')
                    return render_template('upload.html', year=current_year)
                
                # Validation passed
                client.table('jobs').update({'status': 'pending'}).eq('id', job_id).execute()
                logger.info(f'Successful validation for job {job_id}')
                flash('Files uploaded and validated successfully! Please configure the analysis.', 'success')
                return redirect(url_for('configure_analysis', job_id=job_id))
                
            except Exception as e:
                logger.error(f"Error during validation for job {job_id}: {e}")
                client.table('jobs').update({
                    'status': 'failed',
                    'error_message': f'Validation error: {str(e)}'
                }).eq('id', job_id).execute()
                flash(f'Validation error: {e}', 'danger')
                return render_template('upload.html', year=current_year)
                
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}", exc_info=True)
            flash(f'An unexpected error occurred: {e}', 'danger')
            return render_template('upload.html', year=current_year)

    # For GET request, just render the upload form
    return render_template('upload.html', year=current_year)

@app.route('/configure/<job_id>', methods=['GET'])
@login_required
def configure_analysis(job_id):
    """Configuration page for analysis"""
    current_year = datetime.datetime.now().year
    user = get_current_user()
    
    try:
        # Verify job belongs to current user
        client = supabase_client.get_client()
        job_result = client.table('jobs').select('*').eq('id', job_id).eq('user_id', user['id']).execute()
        
        if not job_result.data:
            flash('Job not found or access denied.', 'error')
            return redirect(url_for('index'))
        
        job = job_result.data[0]
        
        # Get job files
        files_result = client.table('job_files').select('*').eq('job_id', job_id).execute()
        job_files = files_result.data if files_result.data else []
        
        return render_template('configure.html', 
                             job_id=job_id, 
                             job=job, 
                             job_files=job_files,
                             year=current_year)
        
    except Exception as e:
        logger.error(f"Error loading configuration for job {job_id}: {e}")
        flash('Error loading job configuration.', 'error')
        return redirect(url_for('index'))

@app.route('/jobs')
@login_required
def list_jobs():
    """List all jobs for current user"""
    current_year = datetime.datetime.now().year
    user = get_current_user()
    
    try:
        client = supabase_client.get_client()
        result = client.table('jobs').select('*').eq('user_id', user['id']).order('created_at', desc=True).execute()
        jobs = result.data if result.data else []
        
        return render_template('jobs.html', jobs=jobs, year=current_year)
        
    except Exception as e:
        logger.error(f"Error fetching jobs for user {user['id']}: {e}")
        flash('Error loading jobs.', 'error')
        return render_template('jobs.html', jobs=[], year=current_year)

@app.route('/about')
def about():
    """Serves a simple about page"""
    current_year = datetime.datetime.now().year
    return f"About Paramaterial - &copy; {current_year}"

if __name__ == '__main__':
    # Test Supabase connection on startup
    if supabase_client.test_connection():
        logger.info("Starting Flask app with Supabase integration")
    else:
        logger.error("Failed to connect to Supabase - check configuration")
    
    app.run(debug=True)