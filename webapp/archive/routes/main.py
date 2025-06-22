from flask import Blueprint, render_template, current_app
from flask_login import login_required, current_user
from models import db, Job, Organization
from datetime import datetime, timedelta

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Landing page"""
    current_year = datetime.now().year
    
    if current_user.is_authenticated:
        return render_template('dashboard.html', year=current_year)
    
    return render_template('landing.html', year=current_year)

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    current_year = datetime.now().year
    
    # Get user's organizations
    organizations = current_user.get_organizations()
    
    # Get recent jobs
    recent_jobs = Job.query.filter_by(user_id=current_user.id)\
                          .order_by(Job.created_at.desc())\
                          .limit(10).all()
    
    # Get job statistics
    total_jobs = Job.query.filter_by(user_id=current_user.id).count()
    completed_jobs = Job.query.filter_by(user_id=current_user.id, status='completed').count()
    failed_jobs = Job.query.filter_by(user_id=current_user.id, status='failed').count()
    processing_jobs = Job.query.filter_by(user_id=current_user.id)\
                                .filter(Job.status.in_(['uploading', 'validating', 'processing'])).count()
    
    # Calculate success rate
    success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    
    stats = {
        'total_jobs': total_jobs,
        'completed_jobs': completed_jobs,
        'failed_jobs': failed_jobs,
        'processing_jobs': processing_jobs,
        'success_rate': round(success_rate, 1)
    }
    
    return render_template('dashboard.html', 
                         year=current_year,
                         organizations=organizations,
                         recent_jobs=recent_jobs,
                         stats=stats)

@main_bp.route('/upload')
@login_required
def upload():
    """File upload page"""
    current_year = datetime.now().year
    
    # Get user's organizations for context
    organizations = current_user.get_organizations()
    
    return render_template('upload.html', 
                         year=current_year,
                         organizations=organizations)

@main_bp.route('/about')
def about():
    """About page"""
    current_year = datetime.now().year
    return render_template('about.html', year=current_year)

@main_bp.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        return {'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}
    except Exception as e:
        current_app.logger.error(f"Health check failed: {e}")
        return {'status': 'unhealthy', 'error': str(e)}, 500