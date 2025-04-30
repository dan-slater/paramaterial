from flask import Flask, render_template, url_for, request, redirect, flash, session, jsonify
import datetime
import os
import uuid
import logging
import traceback
import pandas as pd
import shutil
from werkzeug.utils import secure_filename

from webapp.utils.validation import validate_series_files, extract_ids_from_info_table
from paramaterial.preparing import check_formatting

# --- App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(app.root_path), 'jobs')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.secret_key = os.urandom(24)  # Use a more secure, randomly generated key in production

# Ensure the base jobs directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions (can be refined later)
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    """Check if a file has a valid extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Setup enhanced logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Serves the main dashboard page."""
    current_year = datetime.datetime.now().year
    # For now, the index route renders the dashboard.
    # Later, we can create a separate landing page if needed.
    return render_template('dashboard.html', year=current_year)

@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    """Handles data upload (Info Table and Time Series files)."""
    current_year = datetime.datetime.now().year
    if request.method == 'POST':
        try:
            logger.debug(f"Processing POST request to /upload")
            logger.debug(f"File keys in request: {list(request.files.keys())}")
        except Exception as e:
            logger.error(f"Failed to log request data: {str(e)}")
            
        try:
            # --- File Handling & Initial Validation ---
            # Check Info Table
            if 'info_table' not in request.files:
                logger.warning("No 'info_table' in request.files")
                flash('"Info Table" file part missing in the request.', 'error')
                return render_template('upload.html', year=current_year)
                
            info_file = request.files['info_table']
            logger.debug(f"Info file: {info_file.filename}")
            
            if info_file.filename == '':
                logger.warning("Empty info_table filename")
                flash('No Info Table file selected.', 'error')
                return render_template('upload.html', year=current_year)

            # Check Time Series Files (using getlist)
            series_files = request.files.getlist('time_series_files[]')
            logger.debug(f"Number of time series files: {len(series_files)}")
            for i, f in enumerate(series_files):
                logger.debug(f"Time series file {i}: {f.filename if f else 'None'}")
                
            if not series_files or (series_files[0].filename == ''):
                logger.warning("No time series files provided")
                flash('No Time Series files selected or provided.', 'error')
                return render_template('upload.html', year=current_year)

            # --- Secure Filenames & Create Job Directory ---
            job_id = str(uuid.uuid4())
            job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
            try:
                os.makedirs(job_dir, exist_ok=True)
                logger.debug(f"Created job directory: {job_dir}")
            except Exception as e:
                logger.error(f"Error creating job directory: {str(e)}")
                flash(f'Error creating job directory: {str(e)}', 'error')
                return render_template('upload.html', year=current_year)

            # Save Info Table
            info_filename = secure_filename(info_file.filename)
            info_path = os.path.join(job_dir, info_filename)
            info_file.save(info_path)
            logger.debug(f"Saved info table to: {info_path}")

            # Save Time Series Files
            saved_ts_files = []
            try:
                for ts_file in series_files:
                    if ts_file and ts_file.filename != '': # Ensure file exists
                        ts_filename = secure_filename(ts_file.filename)
                        ts_file_path = os.path.join(job_dir, ts_filename)
                        ts_file.save(ts_file_path)
                        saved_ts_files.append(ts_filename)
                logger.debug(f"Saved {len(saved_ts_files)} time series files")
            except Exception as e:
                logger.error(f"Error saving time series files: {str(e)}\n{traceback.format_exc()}")
                flash(f'Error saving time series files: {str(e)}', 'error')
                # Clean up the job directory if there was an error
                shutil.rmtree(job_dir, ignore_errors=True)
                return render_template('upload.html', year=current_year)

            # --- Validation Logic ---
            try:
                # Extract IDs using the utility function
                expected_ids = extract_ids_from_info_table(info_path)

                # Call validation logic for series files
                is_valid, validation_errors = validate_series_files(job_dir, expected_ids)

                if not is_valid:
                    error_message = "Validation Failed: " + "; ".join(validation_errors)
                    logger.warning(f"Time series validation failed for job {job_id}: {error_message}")
                    # Clean up the job directory
                    shutil.rmtree(job_dir)
                    flash(error_message, 'danger')
                    return render_template('upload.html', year=current_year)
  
            except ValueError as e:
                # Catch ValueErrors raised by extract_ids_from_info_table or other issues
                logger.error(f"Validation error for job {job_id}: {e}", exc_info=True)
                # Clean up the job directory
                shutil.rmtree(job_dir)
                flash(str(e), 'danger') # Display the specific error message
                return render_template('upload.html', year=current_year)
            except Exception as e:
                logger.error(f"Error during validation logic for job {job_id}: {str(e)}", exc_info=True)
                # Clean up the job directory
                shutil.rmtree(job_dir)
                flash(f'An error occurred during validation: {e}', 'danger')
                return render_template('upload.html', year=current_year)

            # If we got here, validation passed
            logger.info(f'Successful validation for job {job_id}. Redirecting to configuration.')
            flash('Files uploaded and validated successfully! Please configure the analysis.', 'success')
            # Redirect to a new configuration page, passing the job_id
            return redirect(url_for('configure_analysis', job_id=job_id))
                
        except Exception as e:
            logger.error(f"An unexpected error occurred during upload processing: {e}", exc_info=True)
            flash(f'An unexpected error occurred: {e}. Please check logs or contact support.', 'danger')
            # Clean up the job directory if it exists and something went wrong early
            if 'job_dir' in locals() and os.path.exists(job_dir):
                 try:
                     shutil.rmtree(job_dir)
                     logger.debug(f'Cleaned up job directory {job_dir} after exception.')
                 except OSError as cleanup_error:
                     logger.error(f'Failed to clean up job directory {job_dir}: {cleanup_error}')
            return render_template('upload.html', year=current_year)

    # For GET request, just render the upload form
    return render_template('upload.html', year=current_year)

@app.route('/about')
def about():
    """Serves a simple about page (placeholder)."""
    current_year = datetime.datetime.now().year
    # You'll need to create an about.html template for this
    # return render_template('about.html', year=current_year)
    return f"About Paramaterial - &copy; {current_year}"

@app.route('/results/<job_id>')
def job_results(job_id):
    """Placeholder route to display results/status for a given job ID."""
    # TODO: Implement logic to fetch job status/results and render a template
    return f"Results page for Job ID: {job_id}. Files uploaded. Next step: Validation and Processing."

@app.route('/configure/<job_id>', methods=['GET'])
def configure_analysis(job_id):
    # TODO: Implement configuration logic
    logger.info(f'Rendering configuration page for job {job_id}')
    current_year = datetime.datetime.now().year
    # We might want to retrieve job details here later based on job_id
    # Render the newly created configure template
    return render_template('configure.html', job_id=job_id, year=current_year)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # TODO: Implement file serving logic
    pass

if __name__ == '__main__':
    # Runs the Flask development server.
    # Debug=True is helpful during development for auto-reloading and error pages.
    # Ensure debug is False in a production environment.
    app.run(debug=True)
