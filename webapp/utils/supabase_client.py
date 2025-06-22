from supabase import create_client, Client
from config import Config
import logging

logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        Config.validate()
        self.url = Config.SUPABASE_URL
        self.key = Config.SUPABASE_KEY
        self.service_key = Config.SUPABASE_SERVICE_KEY
        
        # Create client with anon key for auth operations
        self.client: Client = create_client(self.url, self.key)
        
        # Create service client for admin operations
        if self.service_key:
            self.service_client: Client = create_client(self.url, self.service_key)
        else:
            self.service_client = None
            logger.warning("Service key not provided - some operations may not be available")
    
    def get_client(self, use_service_key=False):
        """Get Supabase client"""
        if use_service_key and self.service_client:
            return self.service_client
        return self.client
    
    def test_connection(self):
        """Test connection to Supabase"""
        try:
            # Try to fetch a single row from auth.users (will fail gracefully if no access)
            result = self.client.table('users').select('id').limit(1).execute()
            logger.info("Supabase connection successful")
            return True
        except Exception as e:
            logger.error(f"Supabase connection failed: {e}")
            return False

# Global instance
supabase_client = SupabaseClient()