from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timedelta
import secrets
from .database import BaseModel, generate_uuid

if TYPE_CHECKING:
    from .user import User
    from .equipment import Equipment
    from .template import AnalysisTemplate
    from .job import Job

class Organization(BaseModel, table=True):
    __tablename__ = 'organizations'
    
    name: str = Field(max_length=200)
    description: Optional[str] = Field(default=None)
    website: Optional[str] = Field(default=None, max_length=255)
    location: Optional[str] = Field(default=None, max_length=100)
    domain: Optional[str] = Field(default=None, max_length=100)  # Optional email domain for auto-suggestions
    logo_url: Optional[str] = Field(default=None, max_length=500)
    # settings: Optional[dict] = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    is_active: bool = Field(default=True)
    
    # Relationships
    memberships: List["OrganizationMembership"] = Relationship(
        back_populates="organization",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    invitations: List["OrganizationInvitation"] = Relationship(
        back_populates="organization",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    equipment: List["Equipment"] = Relationship(
        back_populates="organization",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    templates: List["AnalysisTemplate"] = Relationship(
        back_populates="organization",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    jobs: List["Job"] = Relationship(back_populates="organization")
    
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

class OrganizationMembership(BaseModel, table=True):
    __tablename__ = 'organization_memberships'
    
    organization_id: str = Field(foreign_key="organizations.id")
    user_id: str = Field(foreign_key="users.id")
    role: str = Field(default="member")
    invited_by: Optional[str] = Field(default=None, foreign_key="users.id")
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    organization: Optional["Organization"] = Relationship(back_populates="memberships")
    user: Optional["User"] = Relationship(back_populates="organization_memberships")
    inviter: Optional["User"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "OrganizationMembership.invited_by"}
    )
    
    # TODO: Add unique constraint for (organization_id, user_id)
    
    def is_admin(self) -> bool:
        """Check if member has admin privileges"""
        return self.role in ["owner", "admin"]
    
    def can_invite_members(self) -> bool:
        """Check if member can invite others"""
        return self.role in ["owner", "admin"]
    
    def can_manage_organization(self) -> bool:
        """Check if member can manage organization settings"""
        return self.role == "owner"
    
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

class OrganizationInvitation(BaseModel, table=True):
    __tablename__ = 'organization_invitations'
    
    organization_id: str = Field(foreign_key="organizations.id")
    email: str = Field(max_length=255)
    role: str = Field(default="member")
    invited_by: str = Field(foreign_key="users.id")
    token: str = Field(
        max_length=64,
        unique=True,
        default_factory=lambda: secrets.token_urlsafe(48)
    )
    expires_at: datetime
    is_accepted: bool = Field(default=False)
    accepted_at: Optional[datetime] = Field(default=None)
    accepted_by: Optional[str] = Field(default=None, foreign_key="users.id")
    
    # Relationships
    organization: Optional["Organization"] = Relationship(back_populates="invitations")
    inviter: Optional["User"] = Relationship(
        back_populates="sent_invitations",
        sa_relationship_kwargs={"foreign_keys": "OrganizationInvitation.invited_by"}
    )
    accepter: Optional["User"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "OrganizationInvitation.accepted_by"}
    )
    
    def __init__(self, **kwargs):
        if 'expires_at' not in kwargs:
            # Default expiry is 7 days from creation
            kwargs['expires_at'] = datetime.utcnow() + timedelta(days=7)
        super().__init__(**kwargs)
    
    @property
    def is_expired(self) -> bool:
        """Check if invitation is expired"""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
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