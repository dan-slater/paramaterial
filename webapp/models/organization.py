from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .database import Base, UUIDMixin, TimestampMixin, generate_uuid
import secrets
from datetime import datetime, timedelta

class Organization(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'organizations'
    
    name = Column(String(200), nullable=False)
    description = Column(Text)
    website = Column(String(255))
    location = Column(String(100))
    domain = Column(String(100))  # Optional email domain for auto-suggestions
    logo_url = Column(String(500))
    settings = Column(JSONB, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    memberships = relationship('OrganizationMembership', back_populates='organization', cascade='all, delete-orphan')
    invitations = relationship('OrganizationInvitation', back_populates='organization', cascade='all, delete-orphan')
    equipment = relationship('Equipment', back_populates='organization', cascade='all, delete-orphan')
    templates = relationship('AnalysisTemplate', back_populates='organization', cascade='all, delete-orphan')
    jobs = relationship('Job', back_populates='organization')
    
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
        return membership
    
    def has_member(self, user_id):
        """Check if user is member"""
        return any(m.user_id == user_id for m in self.memberships)
    
    def get_user_role(self, user_id):
        """Get user's role in organization"""
        membership = next((m for m in self.memberships if m.user_id == user_id), None)
        return membership.role if membership else None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'website': self.website,
            'location': self.location,
            'domain': self.domain,
            'logo_url': self.logo_url,
            'is_active': self.is_active,
            'member_count': self.get_member_count(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Organization {self.name}>'

class OrganizationMembership(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'organization_memberships'
    
    organization_id = Column(UUID(as_uuid=False), ForeignKey('organizations.id'), nullable=False)
    user_id = Column(UUID(as_uuid=False), ForeignKey('users.id'), nullable=False)
    role = Column(String(20), nullable=False, default='member')  # owner, admin, member, viewer
    invited_by = Column(UUID(as_uuid=False), ForeignKey('users.id'))
    joined_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    organization = relationship('Organization', back_populates='memberships')
    user = relationship('User', back_populates='organization_memberships')
    inviter = relationship('User', foreign_keys=[invited_by])
    
    # Unique constraint to prevent duplicate memberships
    __table_args__ = (UniqueConstraint('organization_id', 'user_id'),)
    
    def is_admin(self):
        """Check if member has admin privileges"""
        return self.role in ['owner', 'admin']
    
    def can_invite_members(self):
        """Check if member can invite others"""
        return self.role in ['owner', 'admin']
    
    def can_manage_organization(self):
        """Check if member can manage organization settings"""
        return self.role == 'owner'
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'role': self.role,
            'invited_by': self.invited_by,
            'joined_at': self.joined_at.isoformat() if self.joined_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<OrganizationMembership {self.user_id} -> {self.organization_id} ({self.role})>'

class OrganizationInvitation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = 'organization_invitations'
    
    organization_id = Column(UUID(as_uuid=False), ForeignKey('organizations.id'), nullable=False)
    email = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default='member')
    invited_by = Column(UUID(as_uuid=False), ForeignKey('users.id'), nullable=False)
    token = Column(String(64), unique=True, nullable=False, default=lambda: secrets.token_urlsafe(48))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_accepted = Column(Boolean, default=False, nullable=False)
    accepted_at = Column(DateTime(timezone=True))
    accepted_by = Column(UUID(as_uuid=False), ForeignKey('users.id'))
    
    # Relationships
    organization = relationship('Organization', back_populates='invitations')
    inviter = relationship('User', foreign_keys=[invited_by], back_populates='sent_invitations')
    accepter = relationship('User', foreign_keys=[accepted_by])
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.expires_at:
            # Default expiry is 7 days from creation
            self.expires_at = datetime.utcnow() + timedelta(days=7)
    
    @property
    def is_expired(self):
        """Check if invitation is expired"""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self):
        """Check if invitation is valid (not expired and not accepted)"""
        return not self.is_expired and not self.is_accepted
    
    def accept(self, user):
        """Accept invitation"""
        if not self.is_valid:
            raise ValueError("Invitation is no longer valid")
        
        self.is_accepted = True
        self.accepted_at = datetime.utcnow()
        self.accepted_by = user.id
        
        # Create membership
        membership = OrganizationMembership(
            organization_id=self.organization_id,
            user_id=user.id,
            role=self.role,
            invited_by=self.invited_by
        )
        return membership
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'email': self.email,
            'role': self.role,
            'invited_by': self.invited_by,
            'token': self.token,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_accepted': self.is_accepted,
            'accepted_at': self.accepted_at.isoformat() if self.accepted_at else None,
            'accepted_by': self.accepted_by,
            'is_expired': self.is_expired,
            'is_valid': self.is_valid,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<OrganizationInvitation {self.email} -> {self.organization_id} ({self.role})>'