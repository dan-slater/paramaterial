from sqlmodel import SQLModel, Field, Relationship
from passlib.context import CryptContext
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from .database import BaseModel

if TYPE_CHECKING:
    from .organization import OrganizationMembership, OrganizationInvitation
    from .job import Job
    from .template import AnalysisTemplate

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel, table=True):
    __tablename__ = 'users'
    
    email: str = Field(max_length=255, unique=True, index=True)
    password_hash: str = Field(max_length=255)
    first_name: Optional[str] = Field(default=None, max_length=100)
    last_name: Optional[str] = Field(default=None, max_length=100)
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)
    last_login: Optional[datetime] = Field(default=None)
    
    # Relationships
    organization_memberships: List["OrganizationMembership"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    jobs: List["Job"] = Relationship(
        back_populates="user", 
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    created_templates: List["AnalysisTemplate"] = Relationship(
        back_populates="creator",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    sent_invitations: List["OrganizationInvitation"] = Relationship(
        back_populates="inviter",
        sa_relationship_kwargs={
            "foreign_keys": "OrganizationInvitation.invited_by",
            "cascade": "all, delete-orphan"
        }
    )
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = pwd_context.hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return pwd_context.verify(password, self.password_hash)
    
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