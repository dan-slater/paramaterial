# ParaMaterial Web App

A Flask-based MVP for uploading, processing, and visualizing mechanical test (stress-strain) data using the ParaMaterial library.

## Features (MVP)
- Upload CSV/XLSX files
- Process and visualize stress-strain data
- Download results and logs
- Traceable, test-driven, and robust

## Quick Start (Development)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app.py
   ```

## Project Structure
- `app.py` - Flask entry point
- `processing/` - Data processing logic (ParaMaterial integration)
- `static/` - Static files (CSS, JS)
- `templates/` - HTML templates
- `tests/` - Pytest-based tests

## Docker
See Dockerfile for containerization instructions.

## License
MIT (or project default)
