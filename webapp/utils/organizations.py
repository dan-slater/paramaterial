from typing import List, Dict, Optional, Tuple
import logging
from .supabase_client import supabase_client

logger = logging.getLogger(__name__)

class OrganizationManager:
    def __init__(self):
        self.client = supabase_client.get_client()
    
    def create_organization(self, user_id: str, name: str, description: str = None, domain: str = None) -> Dict:
        """Create a new organization"""
        try:
            # Call the database function
            result = self.client.rpc('create_organization', {
                'org_name': name,
                'org_description': description,
                'org_domain': domain
            }).execute()
            
            if result.data:
                org_id = result.data
                logger.info(f"Created organization {name} with ID {org_id}")
                
                # Return the full organization details
                org_result = self.client.table('organizations').select('*').eq('id', org_id).execute()
                return org_result.data[0] if org_result.data else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating organization: {e}")
            raise
    
    def get_user_organizations(self, user_id: str) -> List[Dict]:
        """Get all organizations a user belongs to"""
        try:
            result = self.client.table('organization_memberships').select(
                'role, joined_at, organizations(*)'
            ).eq('user_id', user_id).execute()
            
            organizations = []
            for membership in result.data:
                org_data = membership['organizations']
                org_data['user_role'] = membership['role']
                org_data['joined_at'] = membership['joined_at']
                organizations.append(org_data)
            
            return organizations
            
        except Exception as e:
            logger.error(f"Error fetching user organizations: {e}")
            return []
    
    def get_organization_members(self, org_id: str, user_id: str) -> List[Dict]:
        """Get all members of an organization (requires membership)"""
        try:
            # Check if user is member of organization
            membership_check = self.client.table('organization_memberships').select('role').eq(
                'organization_id', org_id
            ).eq('user_id', user_id).execute()
            
            if not membership_check.data:
                raise PermissionError("User is not a member of this organization")
            
            # Get all members
            result = self.client.table('organization_memberships').select(
                'role, joined_at, invited_by, users(id, email)'
            ).eq('organization_id', org_id).order('joined_at').execute()
            
            members = []
            for membership in result.data:
                user_data = membership['users']
                member_info = {
                    'user_id': user_data['id'],
                    'email': user_data['email'],
                    'role': membership['role'],
                    'joined_at': membership['joined_at'],
                    'invited_by': membership['invited_by']
                }
                members.append(member_info)
            
            return members
            
        except Exception as e:
            logger.error(f"Error fetching organization members: {e}")
            raise
    
    def send_invitation(self, user_id: str, org_id: str, email: str, role: str = 'member', message: str = None) -> str:
        """Send an organization invitation"""
        try:
            result = self.client.rpc('send_organization_invitation', {
                'org_id': org_id,
                'invite_email': email,
                'invite_role': role,
                'invite_message': message
            }).execute()
            
            if result.data:
                invitation_id = result.data
                logger.info(f"Sent invitation to {email} for organization {org_id}")
                
                # Get the invitation token for email/notification
                token_result = self.client.table('organization_invitations').select('token').eq(
                    'id', invitation_id
                ).execute()
                
                return token_result.data[0]['token'] if token_result.data else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error sending invitation: {e}")
            raise
    
    def get_pending_invitations(self, user_id: str) -> List[Dict]:
        """Get user's pending organization invitations"""
        try:
            result = self.client.rpc('get_user_invitations').execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error fetching pending invitations: {e}")
            return []
    
    def accept_invitation(self, user_id: str, token: str) -> bool:
        """Accept an organization invitation"""
        try:
            result = self.client.rpc('accept_organization_invitation', {
                'invitation_token': token
            }).execute()
            
            if result.data:
                logger.info(f"User {user_id} accepted invitation with token {token}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error accepting invitation: {e}")
            raise
    
    def decline_invitation(self, user_id: str, token: str) -> bool:
        """Decline an organization invitation"""
        try:
            result = self.client.rpc('decline_organization_invitation', {
                'invitation_token': token
            }).execute()
            
            if result.data:
                logger.info(f"User {user_id} declined invitation with token {token}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error declining invitation: {e}")
            raise
    
    def get_organization_equipment(self, org_id: str, user_id: str) -> List[Dict]:
        """Get equipment for an organization"""
        try:
            # Verify user is member
            membership_check = self.client.table('organization_memberships').select('role').eq(
                'organization_id', org_id
            ).eq('user_id', user_id).execute()
            
            if not membership_check.data:
                raise PermissionError("User is not a member of this organization")
            
            result = self.client.table('equipment').select('*').eq(
                'organization_id', org_id
            ).eq('is_active', True).order('name').execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error fetching organization equipment: {e}")
            raise
    
    def get_organization_templates(self, org_id: str, user_id: str, equipment_id: str = None) -> List[Dict]:
        """Get analysis templates for an organization"""
        try:
            # Verify user is member
            membership_check = self.client.table('organization_memberships').select('role').eq(
                'organization_id', org_id
            ).eq('user_id', user_id).execute()
            
            if not membership_check.data:
                raise PermissionError("User is not a member of this organization")
            
            query = self.client.table('analysis_templates').select(
                '*, equipment(name, model), users(email)'
            ).eq('organization_id', org_id).eq('is_public', True)
            
            if equipment_id:
                query = query.eq('equipment_id', equipment_id)
            
            result = query.order('usage_count', desc=True).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error fetching organization templates: {e}")
            raise
    
    def create_template(self, user_id: str, org_id: str, equipment_id: str, name: str, 
                       description: str, template_type: str, parameters: Dict, 
                       is_public: bool = False, tags: List[str] = None) -> Dict:
        """Create a new analysis template"""
        try:
            template_data = {
                'organization_id': org_id,
                'equipment_id': equipment_id,
                'created_by': user_id,
                'name': name,
                'description': description,
                'template_type': template_type,
                'parameters': parameters,
                'is_public': is_public,
                'tags': tags or []
            }
            
            result = self.client.table('analysis_templates').insert(template_data).execute()
            
            if result.data:
                logger.info(f"Created template {name} for organization {org_id}")
                return result.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            raise
    
    def use_template(self, user_id: str, template_id: str, job_id: str) -> bool:
        """Record template usage and increment usage count"""
        try:
            # Record usage
            usage_data = {
                'template_id': template_id,
                'job_id': job_id,
                'user_id': user_id
            }
            
            usage_result = self.client.table('template_usage').insert(usage_data).execute()
            
            # Increment usage count
            self.client.rpc('increment', {
                'table_name': 'analysis_templates',
                'row_id': template_id,
                'column_name': 'usage_count'
            }).execute()
            
            return bool(usage_result.data)
            
        except Exception as e:
            logger.error(f"Error recording template usage: {e}")
            return False
    
    def get_organization_stats(self, org_id: str, user_id: str) -> Dict:
        """Get organization statistics"""
        try:
            # Verify user is member
            membership_check = self.client.table('organization_memberships').select('role').eq(
                'organization_id', org_id
            ).eq('user_id', user_id).execute()
            
            if not membership_check.data:
                raise PermissionError("User is not a member of this organization")
            
            # Get member count
            members_result = self.client.table('organization_memberships').select(
                'id', count='exact'
            ).eq('organization_id', org_id).execute()
            
            # Get equipment count
            equipment_result = self.client.table('equipment').select(
                'id', count='exact'
            ).eq('organization_id', org_id).eq('is_active', True).execute()
            
            # Get template count
            templates_result = self.client.table('analysis_templates').select(
                'id', count='exact'
            ).eq('organization_id', org_id).eq('is_public', True).execute()
            
            # Get recent jobs count
            jobs_result = self.client.table('jobs').select(
                'id', count='exact'
            ).eq('organization_id', org_id).execute()
            
            return {
                'member_count': members_result.count or 0,
                'equipment_count': equipment_result.count or 0,
                'template_count': templates_result.count or 0,
                'job_count': jobs_result.count or 0
            }
            
        except Exception as e:
            logger.error(f"Error fetching organization stats: {e}")
            return {}

# Global instance
organization_manager = OrganizationManager()