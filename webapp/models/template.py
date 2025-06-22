from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from .database import db, UUIDMixin, TimestampMixin

class AnalysisTemplate(UUIDMixin, TimestampMixin, db.Model):
    __tablename__ = 'analysis_templates'
    
    organization_id = db.Column(UUID(as_uuid=False), db.ForeignKey('organizations.id'), nullable=False)
    equipment_id = db.Column(UUID(as_uuid=False), db.ForeignKey('equipment.id'))
    created_by = db.Column(UUID(as_uuid=False), db.ForeignKey('users.id'), nullable=False)
    
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    template_type = db.Column(db.Enum('processing', 'analysis', 'visualization', name='template_types'), 
                             nullable=False)
    parameters = db.Column(JSONB, nullable=False, default=dict)
    is_public = db.Column(db.Boolean, default=False, nullable=False)
    usage_count = db.Column(db.Integer, default=0, nullable=False)
    tags = db.Column(ARRAY(db.String), default=list)
    
    # Template versioning
    version = db.Column(db.Integer, default=1, nullable=False)
    parent_template_id = db.Column(UUID(as_uuid=False), db.ForeignKey('analysis_templates.id'))
    is_latest_version = db.Column(db.Boolean, default=True, nullable=False)
    changelog = db.Column(db.Text)
    
    # Category for organization
    category = db.Column(db.String(100))  # 'heat_treatment', 'microstructure', 'mechanical_testing'
    
    # Relationships
    organization = db.relationship('Organization', back_populates='templates')
    equipment = db.relationship('Equipment', back_populates='templates')
    creator = db.relationship('User', back_populates='created_templates')
    jobs = db.relationship('Job', back_populates='template')
    usage_records = db.relationship('TemplateUsage', back_populates='template', cascade='all, delete-orphan')
    
    # Self-referential relationship for versioning
    parent_template = db.relationship('AnalysisTemplate', remote_side='AnalysisTemplate.id', backref='child_versions')
    
    def increment_usage(self):
        """Increment usage count"""
        self.usage_count += 1
    
    def create_new_version(self, user, parameters, changelog=None):
        """Create a new version of this template"""
        # Mark current version as not latest
        self.is_latest_version = False
        
        # Create new version
        new_version = AnalysisTemplate(
            organization_id=self.organization_id,
            equipment_id=self.equipment_id,
            created_by=user.id,
            name=self.name,
            description=self.description,
            template_type=self.template_type,
            parameters=parameters,
            is_public=self.is_public,
            tags=self.tags,
            version=self.version + 1,
            parent_template_id=self.id,
            is_latest_version=True,
            changelog=changelog,
            category=self.category
        )
        
        db.session.add(new_version)
        return new_version
    
    def get_usage_stats(self, days=30):
        """Get usage statistics"""
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent_usage = len([u for u in self.usage_records if u.used_at >= cutoff])
        unique_users = len(set(u.user_id for u in self.usage_records if u.used_at >= cutoff))
        
        return {
            'total_usage': self.usage_count,
            'recent_usage': recent_usage,
            'unique_users': unique_users
        }
    
    def to_dict(self, include_parameters=False):
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'organization_id': self.organization_id,
            'equipment_id': self.equipment_id,
            'equipment_name': self.equipment.name if self.equipment else None,
            'equipment_type': self.equipment.equipment_type if self.equipment else None,
            'created_by': self.created_by,
            'creator_name': self.creator.full_name if self.creator else None,
            'name': self.name,
            'description': self.description,
            'template_type': self.template_type,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'tags': self.tags,
            'version': self.version,
            'parent_template_id': self.parent_template_id,
            'is_latest_version': self.is_latest_version,
            'changelog': self.changelog,
            'category': self.category,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_parameters:
            data['parameters'] = self.parameters
            
        return data
    
    def __repr__(self):
        return f'<AnalysisTemplate {self.name} v{self.version}>'

class TemplateUsage(UUIDMixin, db.Model):
    __tablename__ = 'template_usage'
    
    template_id = db.Column(UUID(as_uuid=False), db.ForeignKey('analysis_templates.id'), nullable=False)
    job_id = db.Column(UUID(as_uuid=False), db.ForeignKey('jobs.id'), nullable=False)
    user_id = db.Column(UUID(as_uuid=False), db.ForeignKey('users.id'), nullable=False)
    used_at = db.Column(db.DateTime(timezone=True), nullable=False, default=db.func.now())
    
    # Parameters actually used (might differ from template)
    actual_parameters = db.Column(JSONB, default=dict)
    
    # Relationships
    template = db.relationship('AnalysisTemplate', back_populates='usage_records')
    job = db.relationship('Job')
    user = db.relationship('User')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'template_id': self.template_id,
            'template_name': self.template.name if self.template else None,
            'job_id': self.job_id,
            'user_id': self.user_id,
            'user_name': self.user.full_name if self.user else None,
            'used_at': self.used_at.isoformat() if self.used_at else None,
            'actual_parameters': self.actual_parameters
        }
    
    def __repr__(self):
        return f'<TemplateUsage {self.template.name} by {self.user.email}>'

# Template categories for UCT Centre for Materials Engineering
TEMPLATE_CATEGORIES = {
    'heat_treatment': {
        'name': 'Heat Treatment',
        'description': 'Thermal processing and heat treatment analysis',
        'color': '#ff6b6b',  # Red
        'icon': 'fire',
        'equipment_types': ['gleeble']
    },
    'microstructure': {
        'name': 'Microstructure Analysis',
        'description': 'Microstructural characterization and imaging',
        'color': '#4ecdc4',  # Teal
        'icon': 'microscope',
        'equipment_types': ['sem', 'optical_microscope']
    },
    'mechanical_testing': {
        'name': 'Mechanical Testing',
        'description': 'Mechanical property evaluation',
        'color': '#45b7d1',  # Blue
        'icon': 'test-tube',
        'equipment_types': ['tensile_tester', 'hardness_tester']
    },
    'crystal_analysis': {
        'name': 'Crystal Structure',
        'description': 'Crystal orientation and texture analysis',
        'color': '#96ceb4',  # Green
        'icon': 'diamond',
        'equipment_types': ['ebsd']
    },
    'failure_analysis': {
        'name': 'Failure Analysis',
        'description': 'Fracture and failure mode analysis',
        'color': '#feca57',  # Yellow
        'icon': 'warning',
        'equipment_types': ['sem', 'optical_microscope']
    },
    'additive_manufacturing': {
        'name': 'Additive Manufacturing',
        'description': 'AM process and material characterization',
        'color': '#ff9ff3',  # Pink
        'icon': 'cube',
        'equipment_types': ['sem', 'tensile_tester', 'hardness_tester']
    }
}