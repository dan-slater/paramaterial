import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { Organization } from '../types';
// Note: API functions would need to be added to api.ts

interface OrganizationsState {
  // State
  organizations: Organization[];
  currentOrganization: Organization | null;
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchOrganizations: () => Promise<void>;
  fetchOrganization: (orgId: string) => Promise<void>;
  createOrganization: (data: Partial<Organization>) => Promise<void>;
  updateOrganization: (orgId: string, data: Partial<Organization>) => Promise<void>;
  deleteOrganization: (orgId: string) => Promise<void>;
  setCurrentOrganization: (org: Organization | null) => void;
  clearError: () => void;
}

export const useOrganizationsStore = create<OrganizationsState>()(
  devtools(
    (set, get) => ({
      // Initial state
      organizations: [],
      currentOrganization: null,
      loading: false,
      error: null,

      // Actions
      fetchOrganizations: async () => {
        set({ loading: true, error: null }, false, 'organizations/fetchOrganizations/start');
        
        try {
          // TODO: Implement API call when organizations endpoint is ready
          // const organizations = await organizationsAPI.getOrganizations();
          
          // Mock implementation for now
          const organizations: Organization[] = [];
          
          set({ 
            organizations,
            loading: false,
            error: null
          }, false, 'organizations/fetchOrganizations/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to fetch organizations';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'organizations/fetchOrganizations/error');
        }
      },

      fetchOrganization: async (_orgId: string) => {
        set({ loading: true, error: null }, false, 'organizations/fetchOrganization/start');
        
        try {
          // TODO: Implement API call when organizations endpoint is ready
          // const organization = await organizationsAPI.getOrganization(orgId);
          
          // Mock implementation for now
          const organization: Organization | null = null;
          
          set({ 
            currentOrganization: organization,
            loading: false,
            error: null
          }, false, 'organizations/fetchOrganization/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to fetch organization';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'organizations/fetchOrganization/error');
        }
      },

      createOrganization: async (data: Partial<Organization>) => {
        set({ loading: true, error: null }, false, 'organizations/createOrganization/start');
        
        try {
          // TODO: Implement API call when organizations endpoint is ready
          // const organization = await organizationsAPI.createOrganization(data);
          
          // Mock implementation for now
          const organization: Organization = {
            id: Date.now().toString(),
            name: data.name || '',
            description: data.description,
            website: data.website,
            location: data.location,
            created_at: new Date().toISOString(),
            member_count: 1
          };
          
          const { organizations } = get();
          set({ 
            organizations: [organization, ...organizations],
            currentOrganization: organization,
            loading: false,
            error: null
          }, false, 'organizations/createOrganization/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to create organization';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'organizations/createOrganization/error');
        }
      },

      updateOrganization: async (orgId: string, data: Partial<Organization>) => {
        set({ loading: true, error: null }, false, 'organizations/updateOrganization/start');
        
        try {
          // TODO: Implement API call when organizations endpoint is ready
          // const organization = await organizationsAPI.updateOrganization(orgId, data);
          
          // Mock implementation for now
          const { organizations, currentOrganization } = get();
          const updatedOrganizations = organizations.map(org => 
            org.id === orgId ? { ...org, ...data, updated_at: new Date().toISOString() } : org
          );
          
          const updatedCurrentOrganization = currentOrganization?.id === orgId
            ? { ...currentOrganization, ...data, updated_at: new Date().toISOString() }
            : currentOrganization;
          
          set({ 
            organizations: updatedOrganizations,
            currentOrganization: updatedCurrentOrganization,
            loading: false,
            error: null
          }, false, 'organizations/updateOrganization/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to update organization';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'organizations/updateOrganization/error');
        }
      },

      deleteOrganization: async (orgId: string) => {
        set({ loading: true, error: null }, false, 'organizations/deleteOrganization/start');
        
        try {
          // TODO: Implement API call when organizations endpoint is ready
          // await organizationsAPI.deleteOrganization(orgId);
          
          const { organizations } = get();
          const updatedOrganizations = organizations.filter(org => org.id !== orgId);
          
          set({ 
            organizations: updatedOrganizations,
            currentOrganization: get().currentOrganization?.id === orgId ? null : get().currentOrganization,
            loading: false,
            error: null
          }, false, 'organizations/deleteOrganization/success');
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Failed to delete organization';
          set({ 
            loading: false, 
            error: errorMessage 
          }, false, 'organizations/deleteOrganization/error');
        }
      },

      setCurrentOrganization: (org: Organization | null) => {
        set({ currentOrganization: org }, false, 'organizations/setCurrentOrganization');
      },

      clearError: () => {
        set({ error: null }, false, 'organizations/clearError');
      },
    }),
    {
      name: 'organizations-store',
    }
  )
);