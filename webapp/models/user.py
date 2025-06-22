from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import UUID
from .database import db, UUIDMixin, TimestampMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, UUIDMixin, TimestampMixin, db.Model):
    __tablename__ = 'users'
    
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    last_login = db.Column(db.DateTime(timezone=True))
    
    # Relationships
    organization_memberships = db.relationship('OrganizationMembership', back_populates='user', cascade='all, delete-orphan')
    jobs = db.relationship('Job', back_populates='user', cascade='all, delete-orphan')
    created_templates = db.relationship('AnalysisTemplate', back_populates='creator', cascade='all, delete-orphan')
    sent_invitations = db.relationship('OrganizationInvitation', foreign_keys='OrganizationInvitation.invited_by', back_populates='inviter')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    @property
    def full_name(self):
        """Get full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.email.split('@')[0]
    
    def get_organizations(self):
        """Get organizations user belongs to"""
        return [membership.organization for membership in self.organization_memberships]
    
    def is_member_of(self, organization_id):
        """Check if user is member of organization"""
        return any(m.organization_id == organization_id for m in self.organization_memberships)
    
    def get_role_in_organization(self, organization_id):
        """Get user's role in specific organization"""
        membership = next((m for m in self.organization_memberships if m.organization_id == organization_id), None)
        return membership.role if membership else None
    
    def can_admin_organization(self, organization_id):
        """Check if user can admin organization"""
        role = self.get_role_in_organization(organization_id)
        return role in ['owner', 'admin']
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.email}>'