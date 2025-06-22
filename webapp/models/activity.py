from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from .database import db, UUIDMixin, TimestampMixin

class ActivityLog(UUIDMixin, db.Model):
    __tablename__ = 'activity_log'
    
    user_id = db.Column(UUID(as_uuid=False), db.ForeignKey('users.id'))
    organization_id = db.Column(UUID(as_uuid=False), db.ForeignKey('organizations.id'))
    
    action_type = db.Column(db.String(100), nullable=False)  # 'job_created', 'template_shared', 'invitation_sent'
    resource_type = db.Column(db.String(50))  # 'job', 'template', 'organization', 'user'
    resource_id = db.Column(UUID(as_uuid=False))
    
    details = db.Column(JSONB, default=dict)
    ip_address = db.Column(INET)
    user_agent = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=db.func.now())
    
    # Relationships
    user = db.relationship('User')
    organization = db.relationship('Organization')
    
    @classmethod
    def log_activity(cls, user_id, action_type, resource_type=None, resource_id=None, 
                     organization_id=None, details=None, ip_address=None, user_agent=None):
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
        db.session.add(activity)
        return activity
    
    @classmethod
    def get_user_activity(cls, user_id, limit=50):
        """Get recent activity for a user"""
        return cls.query.filter_by(user_id=user_id).order_by(cls.created_at.desc()).limit(limit).all()
    
    @classmethod
    def get_organization_activity(cls, organization_id, limit=50):
        """Get recent activity for an organization"""
        return cls.query.filter_by(organization_id=organization_id).order_by(cls.created_at.desc()).limit(limit).all()
    
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
            'ip_address': str(self.ip_address) if self.ip_address else None,
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