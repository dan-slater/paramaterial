from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User
from models.activity import ActivityLog
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))
        
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('auth/login.html')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated. Please contact support.', 'error')
                return render_template('auth/login.html')
            
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            
            # Log activity
            ActivityLog.log_activity(
                user_id=user.id,
                action_type='user_login',
                ip_address=request.environ.get('REMOTE_ADDR'),
                user_agent=request.headers.get('User-Agent')
            )
            
            db.session.commit()
            
            flash(f'Welcome back, {user.full_name}!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid email or password.', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        
        # Validation
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('auth/register.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('auth/register.html')
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'error')
            return render_template('auth/register.html')
        
        # Create new user
        user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            is_verified=True  # Auto-verify for now
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Log activity
        ActivityLog.log_activity(
            user_id=user.id,
            action_type='user_registered',
            ip_address=request.environ.get('REMOTE_ADDR'),
            user_agent=request.headers.get('User-Agent')
        )
        db.session.commit()
        
        login_user(user)
        flash('Registration successful! Welcome to ParaMaterial.', 'success')
        return redirect(url_for('main.dashboard'))
    
    return render_template('auth/register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    # Log activity
    ActivityLog.log_activity(
        user_id=current_user.id,
        action_type='user_logout',
        ip_address=request.environ.get('REMOTE_ADDR'),
        user_agent=request.headers.get('User-Agent')
    )
    db.session.commit()
    
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))