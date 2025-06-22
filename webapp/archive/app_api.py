from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, get_jwt_identity, jwt_required, get_jwt
from flask_restx import Api, Resource
from datetime import timedelta, datetime
import redis
import os
import logging

from config_api import config
from models import db
from api.auth import auth_ns
from api.organizations import organizations_ns  
from api.jobs import jobs_ns
from api.equipment import equipment_ns
from api.templates import templates_ns

def create_app(config_name=None):
    """Application factory for API-only Flask app"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    
    # Setup CORS for React frontend
    CORS(app, origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative React port
        "https://paramaterial.vercel.app",  # Production frontend
    ], supports_credentials=True)
    
    # Setup JWT
    jwt = JWTManager(app)
    
    # Redis for JWT blacklist
    redis_client = redis.from_url(app.config['REDIS_URL'])
    
    # JWT blacklist helpers
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        """Check if JWT token is revoked"""
        jti = jwt_payload['jti']
        token_in_redis = redis_client.get(jti)
        return token_in_redis is not None
    
    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        """Return message when token is revoked"""
        return jsonify({
            'error': 'token_revoked',
            'message': 'The token has been revoked'
        }), 401
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        """Return message when token is expired"""
        return jsonify({
            'error': 'token_expired', 
            'message': 'The token has expired'
        }), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        """Return message when token is invalid"""
        return jsonify({
            'error': 'invalid_token',
            'message': 'Invalid token'
        }), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        """Return message when token is missing"""
        return jsonify({
            'error': 'missing_token',
            'message': 'Authorization token is required'
        }), 401
    
    # Setup Flask-RESTX API
    api = Api(
        app,
        version='1.0',
        title='ParaMaterial API',
        description='Materials testing data parameterization platform',
        doc='/docs/',
        prefix='/api/v1'
    )
    
    # Register namespaces
    api.add_namespace(auth_ns, path='/auth')
    api.add_namespace(organizations_ns, path='/organizations')
    api.add_namespace(jobs_ns, path='/jobs')
    api.add_namespace(equipment_ns, path='/equipment')
    api.add_namespace(templates_ns, path='/templates')
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'not_found',
            'message': 'Resource not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({
            'error': 'internal_server_error',
            'message': 'An internal server error occurred'
        }), 500
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'bad_request',
            'message': 'Bad request'
        }), 400
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        try:
            # Test database connection
            db.session.execute('SELECT 1')
            # Test Redis connection
            redis_client.ping()
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            })
        except Exception as e:
            app.logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    
    # Root endpoint info
    @app.route('/')
    def root():
        """API root endpoint"""
        return jsonify({
            'message': 'ParaMaterial API',
            'version': '1.0.0',
            'documentation': '/docs/',
            'health': '/health'
        })
    
    # Store redis client in app context
    app.redis = redis_client
    
    # Setup logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = logging.FileHandler('logs/paramaterial_api.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('ParaMaterial API startup')
    
    # Initialize config
    config[config_name].init_app(app)
    
    return app

# Create app instance
app = create_app()

# Create tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5555))
    app.run(host='0.0.0.0', port=port, debug=True)