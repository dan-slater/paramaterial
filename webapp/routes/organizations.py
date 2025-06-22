from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models import db, Organization, OrganizationMembership, OrganizationInvitation, Equipment, AnalysisTemplate
from models.activity import ActivityLog
from datetime import datetime

org_bp = Blueprint('organizations', __name__)

@org_bp.route('/')
@login_required
def list_organizations():
    """List user's organizations"""
    organizations = current_user.get_organizations()
    return render_template('organizations/list.html', organizations=organizations)

@org_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create_organization():
    """Create new organization"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        domain = request.form.get('domain', '').strip()
        
        if not name:
            flash('Organization name is required.', 'error')
            return render_template('organizations/create.html')
        
        # Check if organization name already exists
        if Organization.query.filter_by(name=name).first():
            flash('An organization with this name already exists.', 'error')
            return render_template('organizations/create.html')
        
        # Create organization
        org = Organization(
            name=name,
            description=description,
            domain=domain
        )
        db.session.add(org)
        db.session.flush()  # Get the ID
        
        # Add creator as owner
        membership = OrganizationMembership(
            organization_id=org.id,
            user_id=current_user.id,
            role='owner'
        )
        db.session.add(membership)
        
        # Log activity
        ActivityLog.log_activity(
            user_id=current_user.id,
            organization_id=org.id,
            action_type='organization_created',
            resource_type='organization',
            resource_id=org.id,
            details={'name': name}
        )
        
        db.session.commit()
        
        flash(f'Organization "{name}" created successfully!', 'success')
        return redirect(url_for('organizations.view_organization', org_id=org.id))
    
    return render_template('organizations/create.html')

@org_bp.route('/<org_id>')
@login_required
def view_organization(org_id):
    """View organization details"""
    # Check if user is member
    membership = OrganizationMembership.query.filter_by(
        organization_id=org_id,
        user_id=current_user.id
    ).first()
    
    if not membership:
        flash('You do not have access to this organization.', 'error')
        return redirect(url_for('organizations.list_organizations'))
    
    org = Organization.query.get_or_404(org_id)
    
    # Get organization statistics
    members = OrganizationMembership.query.filter_by(organization_id=org_id).all()
    equipment_count = Equipment.query.filter_by(organization_id=org_id, is_active=True).count()
    template_count = AnalysisTemplate.query.filter_by(organization_id=org_id, is_public=True).count()
    
    # Get recent activity
    recent_activity = ActivityLog.get_organization_activity(org_id, limit=10)
    
    return render_template('organizations/view.html',
                         organization=org,
                         membership=membership,
                         members=members,
                         equipment_count=equipment_count,
                         template_count=template_count,
                         recent_activity=recent_activity)

@org_bp.route('/<org_id>/invite', methods=['GET', 'POST'])
@login_required
def invite_member(org_id):
    """Invite new member to organization"""
    # Check if user can invite
    membership = OrganizationMembership.query.filter_by(
        organization_id=org_id,
        user_id=current_user.id
    ).first()
    
    if not membership or not membership.can_invite_members():
        flash('You do not have permission to invite members.', 'error')
        return redirect(url_for('organizations.view_organization', org_id=org_id))
    
    org = Organization.query.get_or_404(org_id)
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        role = request.form.get('role', 'member')
        message = request.form.get('message', '').strip()
        
        if not email:
            flash('Email is required.', 'error')
            return render_template('organizations/invite.html', organization=org)
        
        if role not in ['admin', 'member', 'viewer']:
            flash('Invalid role selected.', 'error')
            return render_template('organizations/invite.html', organization=org)
        
        # Check if user is already a member
        existing_member = OrganizationMembership.query.join(User).filter(
            OrganizationMembership.organization_id == org_id,
            User.email == email
        ).first()
        
        if existing_member:
            flash('This user is already a member of the organization.', 'error')
            return render_template('organizations/invite.html', organization=org)
        
        # Check if invitation already exists
        existing_invitation = OrganizationInvitation.query.filter_by(
            organization_id=org_id,
            email=email
        ).first()
        
        if existing_invitation and existing_invitation.is_pending:
            flash('An invitation has already been sent to this email address.', 'error')
            return render_template('organizations/invite.html', organization=org)
        
        # Delete expired invitation if exists
        if existing_invitation and existing_invitation.is_expired:
            db.session.delete(existing_invitation)
        
        # Create invitation
        invitation = OrganizationInvitation(
            organization_id=org_id,
            email=email,
            role=role,
            invited_by=current_user.id,
            message=message
        )
        db.session.add(invitation)
        
        # Log activity
        ActivityLog.log_activity(
            user_id=current_user.id,
            organization_id=org_id,
            action_type='invitation_sent',
            resource_type='invitation',
            resource_id=invitation.id,
            details={'email': email, 'role': role}
        )
        
        db.session.commit()
        
        # TODO: Send email notification
        
        flash(f'Invitation sent to {email}!', 'success')
        return redirect(url_for('organizations.view_organization', org_id=org_id))
    
    return render_template('organizations/invite.html', organization=org)

@org_bp.route('/invitations')
@login_required
def list_invitations():
    """List pending invitations for current user"""
    invitations = OrganizationInvitation.query.filter_by(
        email=current_user.email
    ).filter(
        OrganizationInvitation.accepted_at.is_(None),
        OrganizationInvitation.expires_at > datetime.utcnow()
    ).all()
    
    return render_template('organizations/invitations.html', invitations=invitations)

@org_bp.route('/invitations/<invitation_id>/accept', methods=['POST'])
@login_required
def accept_invitation(invitation_id):
    """Accept organization invitation"""
    invitation = OrganizationInvitation.query.get_or_404(invitation_id)
    
    if invitation.email != current_user.email:
        flash('This invitation is not for your email address.', 'error')
        return redirect(url_for('organizations.list_invitations'))
    
    if invitation.is_expired:
        flash('This invitation has expired.', 'error')
        return redirect(url_for('organizations.list_invitations'))
    
    if invitation.is_accepted:
        flash('This invitation has already been accepted.', 'error')
        return redirect(url_for('organizations.list_invitations'))
    
    try:
        # Accept invitation (creates membership)
        membership = invitation.accept(current_user)
        
        # Log activity
        ActivityLog.log_activity(
            user_id=current_user.id,
            organization_id=invitation.organization_id,
            action_type='invitation_accepted',
            resource_type='invitation',
            resource_id=invitation.id,
            details={'role': invitation.role}
        )
        
        db.session.commit()
        
        flash(f'Successfully joined {invitation.organization.name}!', 'success')
        return redirect(url_for('organizations.view_organization', org_id=invitation.organization_id))
        
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('organizations.list_invitations'))

@org_bp.route('/invitations/<invitation_id>/decline', methods=['POST'])
@login_required
def decline_invitation(invitation_id):
    """Decline organization invitation"""
    invitation = OrganizationInvitation.query.get_or_404(invitation_id)
    
    if invitation.email != current_user.email:
        flash('This invitation is not for your email address.', 'error')
        return redirect(url_for('organizations.list_invitations'))
    
    # Log activity
    ActivityLog.log_activity(
        user_id=current_user.id,
        organization_id=invitation.organization_id,
        action_type='invitation_declined',
        resource_type='invitation',
        resource_id=invitation.id
    )
    
    # Delete invitation
    db.session.delete(invitation)
    db.session.commit()
    
    flash('Invitation declined.', 'info')
    return redirect(url_for('organizations.list_invitations'))