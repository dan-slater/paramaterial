from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, TYPE_CHECKING
from datetime import datetime
from sqlalchemy import func
from .database import BaseModel

if TYPE_CHECKING:
    from .user import User
    from .organization import Organization

class ActivityLog(BaseModel, table=True):
    __tablename__ = 'activity_logs'
    
    user_id: Optional[str] = Field(default=None, foreign_key="users.id")
    organization_id: Optional[str] = Field(default=None, foreign_key="organizations.id")
    
    action_type: str = Field(max_length=100)  # 'job_created', 'template_shared', 'invitation_sent'
    resource_type: Optional[str] = Field(default=None, max_length=50)  # 'job', 'template', 'organization', 'user'
    resource_id: Optional[str] = Field(default=None)
    
    # details: Optional[dict] = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    ip_address: Optional[str] = Field(default=None, max_length=45)  # IPv6 support
    user_agent: Optional[str] = Field(default=None)
    
    
    # Relationships
    user: Optional["User"] = Relationship()
    organization: Optional["Organization"] = Relationship()
    
    @classmethod
    def log_activity(
        cls,
        user_id: Optional[str],
        action_type: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> "ActivityLog":
        """Log an activity"""
        activity = cls(
            user_id=user_id,
            organization_id=organization_id,
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        return activity
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_name': self.user.full_name if self.user else None,
            'organization_id': self.organization_id,
            'organization_name': self.organization.name if self.organization else None,
            'action_type': self.action_type,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<ActivityLog {self.action_type} by {self.user.email if self.user else "Unknown"}>'

# Common activity types
ACTIVITY_TYPES = {
    # User activities
    'user_registered': 'User registered',
    'user_login': 'User logged in',
    'user_logout': 'User logged out',
    
    # Organization activities
    'organization_created': 'Organization created',
    'organization_updated': 'Organization updated',
    'invitation_sent': 'Invitation sent',
    'invitation_accepted': 'Invitation accepted',
    'invitation_declined': 'Invitation declined',
    'member_added': 'Member added',
    'member_removed': 'Member removed',
    'member_role_changed': 'Member role changed',
    
    # Equipment activities
    'equipment_added': 'Equipment added',
    'equipment_updated': 'Equipment updated',
    'equipment_removed': 'Equipment removed',
    
    # Template activities
    'template_created': 'Template created',
    'template_updated': 'Template updated',
    'template_shared': 'Template shared',
    'template_used': 'Template used',
    'template_version_created': 'Template version created',
    
    # Job activities
    'job_created': 'Job created',
    'job_updated': 'Job updated',
    'job_completed': 'Job completed',
    'job_failed': 'Job failed',
    'file_uploaded': 'File uploaded',
    'file_validated': 'File validated',
    'analysis_started': 'Analysis started',
    'results_exported': 'Results exported',
}