from flask import Flask, render_template, url_for, request, redirect, flash, session, jsonify, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
import redis
import os
import logging
from datetime import datetime

from config_new import config
from models import db, User, Organization, OrganizationMembership, OrganizationInvitation, Job, JobFile, Equipment, AnalysisTemplate
from models.activity import ActivityLog

def create_app(config_name=None):
    """Application factory"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Setup Redis for sessions
    redis_client = redis.from_url(app.config['REDIS_URL'])
    app.config['SESSION_REDIS'] = redis_client
    Session(app)
    
    # Setup Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(user_id)
    
    # Setup logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = logging.FileHandler('logs/paramaterial.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('ParaMaterial startup')
    
    # Register blueprints
    from routes.auth import auth_bp
    from routes.main import main_bp
    from routes.organizations import org_bp
    from routes.jobs import jobs_bp
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(main_bp)
    app.register_blueprint(org_bp, url_prefix='/organizations')
    app.register_blueprint(jobs_bp, url_prefix='/jobs')
    
    # Initialize config
    config[config_name].init_app(app)
    
    return app

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

# Create app instance
app = create_app()

# Create tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5555))
    app.run(host='0.0.0.0', port=port, debug=True)