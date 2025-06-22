from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import UniqueConstraint
from .database import db, UUIDMixin, TimestampMixin, generate_uuid
import secrets
from datetime import datetime, timedelta

class Organization(UUIDMixin, TimestampMixin, db.Model):
    __tablename__ = 'organizations'
    
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    domain = db.Column(db.String(100))  # Optional email domain for auto-suggestions
    logo_url = db.Column(db.String(500))
    settings = db.Column(JSONB, default=dict)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    # Relationships
    memberships = db.relationship('OrganizationMembership', back_populates='organization', cascade='all, delete-orphan')
    invitations = db.relationship('OrganizationInvitation', back_populates='organization', cascade='all, delete-orphan')
    equipment = db.relationship('Equipment', back_populates='organization', cascade='all, delete-orphan')
    templates = db.relationship('AnalysisTemplate', back_populates='organization', cascade='all, delete-orphan')
    jobs = db.relationship('Job', back_populates='organization')
    
    def get_members(self):
        """Get all members of organization"""
        return [membership.user for membership in self.memberships]
    
    def get_admins(self):
        """Get admin members"""
        return [m.user for m in self.memberships if m.role in ['owner', 'admin']]
    
    def get_member_count(self):
        """Get count of active members"""
        return len(self.memberships)
    
    def add_member(self, user, role='member', invited_by=None):
        """Add user as member"""
        membership = OrganizationMembership(
            organization_id=self.id,
            user_id=user.id,
            role=role,
            invited_by=invited_by
        )
        db.session.add(membership)
        return membership
    
    def remove_member(self, user):
        """Remove user from organization"""
        membership = OrganizationMembership.query.filter_by(
            organization_id=self.id,
            user_id=user.id
        ).first()
        if membership:
            db.session.delete(membership)
            return True
        return False
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'domain': self.domain,
            'logo_url': self.logo_url,
            'member_count': self.get_member_count(),
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Organization {self.name}>'

class OrganizationMembership(UUIDMixin, db.Model):
    __tablename__ = 'organization_memberships'
    __table_args__ = (UniqueConstraint('organization_id', 'user_id'),)
    
    organization_id = db.Column(UUID(as_uuid=False), db.ForeignKey('organizations.id'), nullable=False)
    user_id = db.Column(UUID(as_uuid=False), db.ForeignKey('users.id'), nullable=False)
    role = db.Column(db.Enum('owner', 'admin', 'member', 'viewer', name='membership_roles'), 
                     nullable=False, default='member')
    joined_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    invited_by = db.Column(UUID(as_uuid=False), db.ForeignKey('users.id'))
    
    # Relationships
    organization = db.relationship('Organization', back_populates='memberships')
    user = db.relationship('User', back_populates='organization_memberships', foreign_keys=[user_id])
    inviter = db.relationship('User', foreign_keys=[invited_by])
    
    def can_invite_members(self):
        """Check if this membership can invite new members"""
        return self.role in ['owner', 'admin']
    
    def can_manage_organization(self):
        """Check if this membership can manage organization"""
        return self.role in ['owner', 'admin']
    
    def can_create_templates(self):
        """Check if this membership can create templates"""
        return self.role in ['owner', 'admin', 'member']
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'role': self.role,
            'joined_at': self.joined_at.isoformat() if self.joined_at else None,
            'invited_by': self.invited_by
        }
    
    def __repr__(self):
        return f'<OrganizationMembership {self.user.email} -> {self.organization.name} ({self.role})>'

class OrganizationInvitation(UUIDMixin, TimestampMixin, db.Model):
    __tablename__ = 'organization_invitations'
    __table_args__ = (UniqueConstraint('organization_id', 'email'),)
    
    organization_id = db.Column(UUID(as_uuid=False), db.ForeignKey('organizations.id'), nullable=False)
    email = db.Column(db.String(255), nullable=False, index=True)
    role = db.Column(db.Enum('admin', 'member', 'viewer', name='invitation_roles'), 
                     nullable=False, default='member')
    invited_by = db.Column(UUID(as_uuid=False), db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text)
    token = db.Column(db.String(255), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    accepted_at = db.Column(db.DateTime(timezone=True))
    accepted_by = db.Column(UUID(as_uuid=False), db.ForeignKey('users.id'))
    
    # Relationships
    organization = db.relationship('Organization', back_populates='invitations')
    inviter = db.relationship('User', foreign_keys=[invited_by], back_populates='sent_invitations')
    accepter = db.relationship('User', foreign_keys=[accepted_by])
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.token:
            self.token = secrets.token_urlsafe(32)
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(days=7)
    
    @property
    def is_expired(self):
        """Check if invitation is expired"""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_accepted(self):
        """Check if invitation is accepted"""
        return self.accepted_at is not None
    
    @property
    def is_pending(self):
        """Check if invitation is pending"""
        return not self.is_expired and not self.is_accepted
    
    def accept(self, user):
        """Accept invitation"""
        if self.is_expired:
            raise ValueError("Invitation has expired")
        if self.is_accepted:
            raise ValueError("Invitation already accepted")
        if user.email != self.email:
            raise ValueError("User email does not match invitation")
        
        self.accepted_at = datetime.utcnow()
        self.accepted_by = user.id
        
        # Create membership
        membership = self.organization.add_member(user, self.role, self.invited_by)
        db.session.add(membership)
        
        return membership
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'organization_name': self.organization.name if self.organization else None,
            'email': self.email,
            'role': self.role,
            'invited_by': self.invited_by,
            'inviter_name': self.inviter.full_name if self.inviter else None,
            'message': self.message,
            'token': self.token,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'accepted_at': self.accepted_at.isoformat() if self.accepted_at else None,
            'is_expired': self.is_expired,
            'is_accepted': self.is_accepted,
            'is_pending': self.is_pending,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<OrganizationInvitation {self.email} -> {self.organization.name}>'