from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class OrganizationRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

class OrganizationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = None
    location: Optional[str] = Field(None, max_length=100)

class OrganizationCreate(OrganizationBase):
    pass

class OrganizationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = None
    location: Optional[str] = Field(None, max_length=100)

class OrganizationResponse(OrganizationBase):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    member_count: Optional[int] = 0
    
    class Config:
        from_attributes = True

class OrganizationMemberResponse(BaseModel):
    id: str
    user_id: str
    organization_id: str
    role: OrganizationRole
    joined_at: datetime
    user_email: str
    user_name: str
    
    class Config:
        from_attributes = True

class OrganizationInviteCreate(BaseModel):
    email: str = Field(..., description="Email address to invite")
    role: OrganizationRole = Field(default=OrganizationRole.MEMBER)
    message: Optional[str] = Field(None, max_length=500)

class OrganizationInviteResponse(BaseModel):
    id: str
    organization_id: str
    email: str
    role: OrganizationRole
    invited_by: str
    created_at: datetime
    expires_at: datetime
    is_accepted: bool
    
    class Config:
        from_attributes = True

class InviteAccept(BaseModel):
    token: str

class MemberRoleUpdate(BaseModel):
    role: OrganizationRole