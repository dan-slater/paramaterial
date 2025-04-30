from flask import Flask, render_template, url_for, request, redirect, flash, session
import datetime
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Use a more secure, randomly generated key in production
app.secret_key = os.urandom(24) 

# Define upload folder relative to the project root (one level up from webapp)
# Use a configuration variable for better practice
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(app.root_path), 'jobs')
# Ensure the base jobs directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions (can be refined later)
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        # --- File Handling Logic ---
        info_file = request.files.get('info_table')
        time_series_files = request.files.getlist('time_series_files[]')

        # Basic validation: Check if files exist
        if not info_file or info_file.filename == '':
            flash('Info Table file is required.', 'error')
            return redirect(request.url)
        if not time_series_files or all(f.filename == '' for f in time_series_files):
            flash('At least one Time Series file is required.', 'error')
            return redirect(request.url)

        # Validate info file extension
        if not allowed_file(info_file.filename):
            flash(f'Invalid file type for Info Table: {info_file.filename}. Allowed: {ALLOWED_EXTENSIONS}', 'error')
            return redirect(request.url)
        
        # Validate time series file extensions
        for ts_file in time_series_files:
            if ts_file and not allowed_file(ts_file.filename):
                 flash(f'Invalid file type for Time Series file: {ts_file.filename}. Allowed: .csv', 'error') # Assuming only CSV for TS
                 # Note: Need to refine ALLOWED_EXTENSIONS check if TS allows only CSV
                 return redirect(request.url)

        # Generate Job ID and create job directory
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        try:
            os.makedirs(job_dir)
            app.logger.info(f"Created job directory: {job_dir}")
        except OSError as e:
            app.logger.error(f"Error creating directory {job_dir}: {e}")
            flash('Failed to create job directory. Please try again.', 'error')
            return redirect(request.url)

        # Save Info Table
        try:
            info_filename = secure_filename(info_file.filename)
            info_file_path = os.path.join(job_dir, info_filename)
            info_file.save(info_file_path)
            app.logger.info(f"Saved Info Table: {info_file_path}")
        except Exception as e:
            app.logger.error(f"Error saving Info Table for job {job_id}: {e}")
            flash('Error saving Info Table file.', 'error')
            # Consider cleanup of job_dir here
            return redirect(request.url)

        # Save Time Series Files
        saved_ts_files = []
        try:
            for ts_file in time_series_files:
                if ts_file and ts_file.filename != '': # Ensure file exists
                    ts_filename = secure_filename(ts_file.filename)
                    ts_file_path = os.path.join(job_dir, ts_filename)
                    ts_file.save(ts_file_path)
                    saved_ts_files.append(ts_filename)
                    app.logger.info(f"Saved Time Series file: {ts_file_path}")
            app.logger.info(f"Saved {len(saved_ts_files)} Time Series files for job {job_id}.")
        except Exception as e:
            app.logger.error(f"Error saving Time Series files for job {job_id}: {e}")
            flash('Error saving one or more Time Series files.', 'error')
            # Consider cleanup of job_dir here
            return redirect(request.url)
            
        flash(f'Files successfully uploaded for Job ID: {job_id}', 'success')
        # Store job_id in session for potential later use
        session['current_job_id'] = job_id 
        # Redirect to a results page (needs to be created)
        return redirect(url_for('job_results', job_id=job_id))

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

if __name__ == '__main__':
    # Runs the Flask development server.
    # Debug=True is helpful during development for auto-reloading and error pages.
    # Ensure debug is False in a production environment.
    app.run(debug=True)
