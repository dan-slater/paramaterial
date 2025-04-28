from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify, send_from_directory
import os
import logging
from werkzeug.utils import secure_filename
from uuid import uuid4
import pandas as pd
import shutil
from .plotting import create_stress_strain_plot

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret')

# Logging setup
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('webapp.log', mode='a')
    ])
logger = logging.getLogger(__name__)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ROUTES
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        job_id = str(uuid4())
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(save_path)
        logger.info(f"File uploaded: {filename} (Job ID: {job_id})")
        # Validate file size
        if os.path.getsize(save_path) > MAX_CONTENT_LENGTH:
            os.remove(save_path)
            flash('File too large (limit: 10MB).')
            return redirect(url_for('index'))
        # Try to parse file and check for columns
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(save_path)
            else:
                df = pd.read_excel(save_path)
        except Exception as e:
            logger.error(f"Failed to parse file: {e}")
            flash('Failed to parse file. Please upload a valid CSV or Excel file.')
            return redirect(url_for('index'))
        # Check for required columns
        columns = [c.lower() for c in df.columns]
        strain_col = next((c for c in columns if 'strain' in c), None)
        stress_col = next((c for c in columns if 'stress' in c), None)
        if not strain_col or not stress_col:
            flash('File must contain columns for Strain and Stress.')
            return redirect(url_for('index'))
        # Save parsed DataFrame for processing (as pickle for MVP simplicity)
        job_dir = os.path.join('jobs', job_id)
        os.makedirs(job_dir, exist_ok=True)
        df.to_pickle(os.path.join(job_dir, 'data.pkl'))
        # Copy original uploaded file into job directory
        try:
            shutil.copy2(save_path, os.path.join(job_dir, filename))
            logger.info(f"Original file {filename} copied to job directory {job_dir}")
        except Exception as e:
            logger.error(f"Failed to copy original file {filename} to {job_dir}: {e}")
            # Optionally flash an error message to the user if copy fails?
            # flash('Error saving original file for browsing.')
        logger.info(f"File validated and parsed for Job ID: {job_id}")
        return redirect(url_for('results', job_id=job_id, file=filename))
    else:
        flash('Invalid file type.')
        return redirect(url_for('index'))

@app.route('/process')
def process():
    job_id = request.args.get('job_id')
    filename = request.args.get('filename')
    if not job_id or not filename:
        flash('Missing job information.')
        return redirect(url_for('index'))
    job_dir = os.path.join('jobs', job_id)
    data_path = os.path.join(job_dir, 'data.pkl')
    if not os.path.exists(data_path):
        flash('No data found for this job.')
        return redirect(url_for('index'))
    df = pd.read_pickle(data_path)
    # For now, just show column names as a placeholder
    logger.info(f"Processing started for Job ID: {job_id}")
    return render_template('results.html', job_id=job_id, filename=filename, results={'columns': list(df.columns)})

@app.route('/results/<job_id>')
def results(job_id):
    job_dir = os.path.join('jobs', job_id)
    if not os.path.exists(job_dir):
        flash('Job not found.')
        return redirect(url_for('index'))
    # List files in job directory
    file_list = []
    try:
        for fname in sorted(os.listdir(job_dir)): # Sort files for consistency
            fpath = os.path.join(job_dir, fname)
            if os.path.isfile(fpath):
                file_list.append(fname)
    except OSError as e:
        logger.error(f"Error listing files in job directory {job_dir}: {e}")
        flash('Error accessing job files.')
        return redirect(url_for('index'))

    logger.info(f"Files listed for job {job_id}: {file_list}") # Log files found

    selected_file = request.args.get('file')
    # If no specific file is requested via URL param, try to select the first file
    if not selected_file and file_list:
        selected_file = file_list[0]
    # If a specific file is requested but not found (e.g., after deletion?), default to first or none
    elif selected_file and selected_file not in file_list:
         logger.warning(f"Requested file '{selected_file}' not found in job {job_id}. Defaulting selection.")
         selected_file = file_list[0] if file_list else None

    preview_data = None
    preview_error = None
    if selected_file:
        preview_result = preview_file(os.path.join(job_dir, selected_file))
        if isinstance(preview_result, dict):
            preview_data = preview_result
        else:
            preview_error = preview_result # Capture error message

    return render_template('results.html', job_id=job_id, file_list=file_list, selected_file=selected_file, preview_data=preview_data, preview_error=preview_error)


def preview_file(fpath, nrows=20):
    try:
        logger.info(f"Attempting to preview file: {fpath}")
        if fpath.lower().endswith('.csv'):
            # Try detecting encoding
            try:
                df = pd.read_csv(fpath, nrows=nrows)
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {fpath}, trying latin1.")
                df = pd.read_csv(fpath, nrows=nrows, encoding='latin1')
        elif fpath.lower().endswith('.xlsx'):
            df = pd.read_excel(fpath, nrows=nrows)
        elif fpath.lower().endswith('.pkl'):
            df = pd.read_pickle(fpath)
            df = df.head(nrows)
        else:
            logger.warning(f"Unsupported file type for preview: {fpath}")
            return "Unsupported file type for preview."
        # Convert NaNs to strings for display
        df = df.fillna('NaN')
        return {'columns': list(df.columns), 'rows': df.values.tolist()}
    except FileNotFoundError:
        logger.error(f'Preview failed: File not found at {fpath}')
        return "File not found."
    except pd.errors.EmptyDataError:
        logger.warning(f'Preview skipped: Empty file {fpath}')
        return "File is empty."
    except Exception as e:
        logger.error(f'Preview failed for file {fpath}: {e}')
        return f"Error reading file: {e}"

@app.route('/results/<job_id>/file/<filename>')
def file_preview(job_id, filename):
    job_dir = os.path.join('jobs', job_id)
    fpath = os.path.join(job_dir, filename)
    preview_data = preview_file(fpath)
    return jsonify(preview_data) if preview_data else ('Could not preview file', 400)


@app.route('/download/<job_id>/<filename>')
def download_file(job_id, filename):
    # Use absolute path for send_from_directory
    job_dir_abs = os.path.join(app.root_path, 'jobs', job_id)
    logging.info(f"Download request for: {os.path.join(job_dir_abs, filename)}")
    try:
        # Ensure the filename is secure and exists within the job directory
        # send_from_directory handles security checks like path traversal
        return send_from_directory(directory=job_dir_abs, path=filename, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"File not found for download: {os.path.join(job_dir_abs, filename)}")
        flash(f"Error: File '{filename}' not found.", 'error')
        # Redirect back to results page, maybe keep the selected file context?
        return redirect(url_for('results', job_id=job_id, file=filename))
    except Exception as e:
        logging.error(f"Error during download of {os.path.join(job_dir_abs, filename)}: {e}")
        flash(f"An unexpected error occurred during download.", 'error')
        return redirect(url_for('results', job_id=job_id, file=filename))

@app.route('/download/<type>/<job_id>')
def download(type, job_id):
    # Placeholder: Serve files for download
    return f"Download {type} for job {job_id} (not implemented)"

# New route to get column names for a file
@app.route('/columns/<job_id>/<filename>')
def get_columns(job_id, filename):
    logging.info(f"Column request received for job_id: {job_id}, filename: {filename}")
    job_dir_abs = os.path.join(app.root_path, 'jobs', job_id)
    file_path = os.path.join(job_dir_abs, filename)
    logging.info(f"Full path for column request: {file_path}")
    logging.info(f"File exists: {os.path.exists(file_path)}")

    if not filename.lower().endswith(('.csv', '.xlsx')):
        logging.warning(f"Column request for non-data file: {filename}")
        return jsonify({"error": "Column listing only supported for CSV/XLSX files."}), 400

    if not os.path.exists(file_path):
        logging.error(f"File not found for column listing: {file_path}")
        return jsonify({"error": f"File '{filename}' not found."}), 404

    try:
        # Read only the header row to get columns efficiently
        logging.info(f"Attempting to read columns from {filename}")
        if filename.lower().endswith('.csv'):
            df_header = pd.read_csv(file_path, nrows=0)
        else: # .xlsx
            df_header = pd.read_excel(file_path, nrows=0)
            
        columns = df_header.columns.tolist()
        logging.info(f"Successfully read columns: {columns}")
        return jsonify({"columns": columns})
    except Exception as e:
        logging.exception(f"Error reading columns for {file_path}: {e}")
        return jsonify({"error": "Could not read columns from file."}), 500

@app.route('/plot/<job_id>/<filename>')
def get_plot_json(job_id, filename):
    # Get column names from query parameters
    x_col = request.args.get('x_col')
    y_col = request.args.get('y_col')

    logging.info(f"Plot request with params - job_id: {job_id}, filename: {filename}, x_col: {x_col}, y_col: {y_col}")

    if not x_col or not y_col:
        logging.error(f"Missing x_col or y_col parameter for plot request: {filename}")
        return jsonify({"error": "Missing required column parameters (x_col, y_col)."}), 400

    job_dir_abs = os.path.join(app.root_path, 'jobs', job_id)
    file_path = os.path.join(job_dir_abs, filename)
    logging.info(f"Plot request path: {file_path} (exists: {os.path.exists(file_path)})")

    if not filename.lower().endswith(('.csv', '.xlsx')):
        logging.warning(f"Plotting attempt on non-data file: {filename}")
        return jsonify({"error": "Plotting only supported for CSV/XLSX files."}), 400

    if not os.path.exists(file_path):
        logging.error(f"File not found for plotting: {file_path}")
        return jsonify({"error": f"File '{filename}' not found."}), 404

    try:
        # Read the entire file for plotting
        df = pd.read_csv(file_path) if filename.lower().endswith('.csv') else pd.read_excel(file_path)

        # Generate plot JSON using specified columns
        plot_json = create_stress_strain_plot(df, x_col, y_col)
        # Return the JSON string directly
        # We manually set content type because Flask's jsonify would re-encode it
        return plot_json, 200, {'Content-Type': 'application/json'}

    except ValueError as e:
        # Errors from create_stress_strain_plot (missing cols, bad data)
        logging.error(f"Value error during plot generation for {file_path}: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # General errors (e.g., file reading issues)
        logging.exception(f"Unexpected error generating plot for {file_path}: {e}") # Log traceback
        return jsonify({"error": "An unexpected error occurred while generating the plot."}), 500


if __name__ == '__main__':
    # Bind to 0.0.0.0 to allow access from proxies/containers
    app.run(host='0.0.0.0', port=5001, debug=True)
