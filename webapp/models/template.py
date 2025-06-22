from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from .database import BaseModel

if TYPE_CHECKING:
    from .organization import Organization
    from .equipment import Equipment
    from .user import User
    from .job import Job

class AnalysisTemplate(BaseModel, table=True):
    __tablename__ = 'analysis_templates'
    
    organization_id: str = Field(foreign_key="organizations.id")
    equipment_id: Optional[str] = Field(default=None, foreign_key="equipment.id")
    created_by: str = Field(foreign_key="users.id")
    
    name: str = Field(max_length=200)
    description: Optional[str] = Field(default=None)
    template_type: str
    # template_data: dict = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    # parameters: dict = Field(default_factory=dict)  # TODO: Add back with proper JSON type
    is_public: bool = Field(default=False)
    usage_count: int = Field(default=0)
    # tags: Optional[List[str]] = Field(default_factory=list)  # TODO: Add back with proper ARRAY type
    
    # Template versioning
    version: int = Field(default=1)
    parent_template_id: Optional[str] = Field(default=None, foreign_key="analysis_templates.id")
    
    # Relationships
    organization: Optional["Organization"] = Relationship(back_populates="templates")
    equipment: Optional["Equipment"] = Relationship(back_populates="templates")
    creator: Optional["User"] = Relationship(back_populates="created_templates")
    jobs: List["Job"] = Relationship(back_populates="template")
    parent_template: Optional["AnalysisTemplate"] = Relationship(
        sa_relationship_kwargs={"remote_side": "AnalysisTemplate.id"}
    )
    
    def increment_usage(self):
        """Increment usage count when template is used"""
        self.usage_count += 1
    
    def is_accessible_by(self, user):
        """Check if template is accessible by user"""
        if self.is_public:
            return True
        if self.created_by == user.id:
            return True
        # Check if user is member of organization
        return user.is_member_of(self.organization_id)
    
    def clone(self, new_name, created_by):
        """Create a clone of this template"""
        clone = AnalysisTemplate(
            organization_id=self.organization_id,
            equipment_id=self.equipment_id,
            created_by=created_by,
            name=new_name,
            description=f"Clone of {self.name}",
            template_type=self.template_type,
            template_data=self.template_data.copy(),
            parameters=self.parameters.copy(),
            is_public=False,
            tags=self.tags.copy() if self.tags else [],
            parent_template_id=self.id
        )
        return clone
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'equipment_id': self.equipment_id,
            'created_by': self.created_by,
            'name': self.name,
            'description': self.description,
            'template_type': self.template_type,
            'template_data': self.template_data,
            'parameters': self.parameters,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'tags': self.tags,
            'version': self.version,
            'parent_template_id': self.parent_template_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<AnalysisTemplate {self.name} ({self.template_type})>'