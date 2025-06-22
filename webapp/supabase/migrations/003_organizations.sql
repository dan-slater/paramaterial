-- Organizations and Team Management
-- Migration for adding organization features to ParaMaterial

-- Create organizations table
CREATE TABLE public.organizations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    domain TEXT, -- e.g., "uct.ac.za" for automatic user assignment
    logo_url TEXT,
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create organization memberships table
CREATE TABLE public.organization_memberships (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    invited_by UUID REFERENCES public.users(id),
    
    UNIQUE(organization_id, user_id)
);

-- Create equipment/machines table
CREATE TABLE public.equipment (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    model TEXT,
    description TEXT,
    equipment_type TEXT NOT NULL, -- 'gleeble', 'sem', 'ebsd', 'tensile_tester', etc.
    specifications JSONB DEFAULT '{}'::jsonb,
    location TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create analysis templates table
CREATE TABLE public.analysis_templates (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE NOT NULL,
    equipment_id UUID REFERENCES public.equipment(id) ON DELETE SET NULL,
    created_by UUID REFERENCES public.users(id) ON DELETE SET NULL NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    template_type TEXT NOT NULL, -- 'processing', 'analysis', 'visualization'
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_public BOOLEAN DEFAULT FALSE, -- visible to all org members
    usage_count INTEGER DEFAULT 0,
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create template usage tracking
CREATE TABLE public.template_usage (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    template_id UUID REFERENCES public.analysis_templates(id) ON DELETE CASCADE NOT NULL,
    job_id UUID REFERENCES public.jobs(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
    used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add organization_id to existing jobs table
ALTER TABLE public.jobs 
ADD COLUMN organization_id UUID REFERENCES public.organizations(id) ON DELETE SET NULL,
ADD COLUMN template_id UUID REFERENCES public.analysis_templates(id) ON DELETE SET NULL,
ADD COLUMN equipment_id UUID REFERENCES public.equipment(id) ON DELETE SET NULL;

-- Create indexes for performance
CREATE INDEX idx_organization_memberships_org_id ON public.organization_memberships(organization_id);
CREATE INDEX idx_organization_memberships_user_id ON public.organization_memberships(user_id);
CREATE INDEX idx_equipment_organization ON public.equipment(organization_id);
CREATE INDEX idx_equipment_type ON public.equipment(equipment_type);
CREATE INDEX idx_templates_organization ON public.analysis_templates(organization_id);
CREATE INDEX idx_templates_equipment ON public.analysis_templates(equipment_id);
CREATE INDEX idx_templates_public ON public.analysis_templates(is_public);
CREATE INDEX idx_template_usage_template ON public.template_usage(template_id);
CREATE INDEX idx_jobs_organization ON public.jobs(organization_id);

-- Enable Row Level Security
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.equipment ENABLE ROW LEVEL SECURITY;
CREATE TABLE public.analysis_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.template_usage ENABLE ROW LEVEL SECURITY;

-- Organization policies
CREATE POLICY "Organization members can view their organizations" ON public.organizations
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = organizations.id 
            AND user_id = auth.uid()
        )
    );

CREATE POLICY "Organization owners can update their organizations" ON public.organizations
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = organizations.id 
            AND user_id = auth.uid() 
            AND role = 'owner'
        )
    );

-- Membership policies
CREATE POLICY "Users can view memberships in their organizations" ON public.organization_memberships
    FOR SELECT USING (
        user_id = auth.uid() OR
        EXISTS (
            SELECT 1 FROM public.organization_memberships AS om2
            WHERE om2.organization_id = organization_memberships.organization_id 
            AND om2.user_id = auth.uid()
        )
    );

CREATE POLICY "Organization admins can manage memberships" ON public.organization_memberships
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.organization_memberships AS om2
            WHERE om2.organization_id = organization_memberships.organization_id 
            AND om2.user_id = auth.uid() 
            AND om2.role IN ('owner', 'admin')
        )
    );

-- Equipment policies
CREATE POLICY "Organization members can view equipment" ON public.equipment
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = equipment.organization_id 
            AND user_id = auth.uid()
        )
    );

CREATE POLICY "Organization admins can manage equipment" ON public.equipment
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = equipment.organization_id 
            AND user_id = auth.uid() 
            AND role IN ('owner', 'admin')
        )
    );

-- Template policies
CREATE POLICY "Organization members can view public templates" ON public.analysis_templates
    FOR SELECT USING (
        is_public = TRUE AND EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = analysis_templates.organization_id 
            AND user_id = auth.uid()
        ) OR created_by = auth.uid()
    );

CREATE POLICY "Users can manage their own templates" ON public.analysis_templates
    FOR ALL USING (created_by = auth.uid());

-- Template usage policies
CREATE POLICY "Users can view their template usage" ON public.template_usage
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can create template usage records" ON public.template_usage
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Update jobs policies to include organization context
DROP POLICY "Users can view own jobs" ON public.jobs;
CREATE POLICY "Users can view own jobs or organization jobs" ON public.jobs
    FOR SELECT USING (
        auth.uid() = user_id OR
        (organization_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = jobs.organization_id 
            AND user_id = auth.uid()
        ))
    );

-- Functions for organization management
CREATE OR REPLACE FUNCTION public.create_organization(
    org_name TEXT,
    org_description TEXT DEFAULT NULL,
    org_domain TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    org_id UUID;
BEGIN
    -- Create organization
    INSERT INTO public.organizations (name, description, domain)
    VALUES (org_name, org_description, org_domain)
    RETURNING id INTO org_id;
    
    -- Add creator as owner
    INSERT INTO public.organization_memberships (organization_id, user_id, role)
    VALUES (org_id, auth.uid(), 'owner');
    
    RETURN org_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create organization invitations table
CREATE TABLE public.organization_invitations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
    invited_by UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
    message TEXT,
    token TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() + INTERVAL '7 days'),
    accepted_at TIMESTAMP WITH TIME ZONE,
    accepted_by UUID REFERENCES public.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(organization_id, email)
);

-- Create invitation notifications table
CREATE TABLE public.invitation_notifications (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
    invitation_id UUID REFERENCES public.organization_invitations(id) ON DELETE CASCADE NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_invitations_organization ON public.organization_invitations(organization_id);
CREATE INDEX idx_invitations_email ON public.organization_invitations(email);
CREATE INDEX idx_invitations_token ON public.organization_invitations(token);
CREATE INDEX idx_invitations_expires ON public.organization_invitations(expires_at);
CREATE INDEX idx_notifications_user ON public.invitation_notifications(user_id);
CREATE INDEX idx_notifications_unread ON public.invitation_notifications(user_id, is_read);

-- Enable RLS
ALTER TABLE public.organization_invitations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.invitation_notifications ENABLE ROW LEVEL SECURITY;

-- Invitation policies
CREATE POLICY "Organization admins can view invitations" ON public.organization_invitations
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = organization_invitations.organization_id 
            AND user_id = auth.uid() 
            AND role IN ('owner', 'admin')
        )
    );

CREATE POLICY "Users can view invitations sent to their email" ON public.organization_invitations
    FOR SELECT USING (
        email = (SELECT email FROM public.users WHERE id = auth.uid())
    );

CREATE POLICY "Organization admins can manage invitations" ON public.organization_invitations
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.organization_memberships 
            WHERE organization_id = organization_invitations.organization_id 
            AND user_id = auth.uid() 
            AND role IN ('owner', 'admin')
        )
    );

-- Notification policies
CREATE POLICY "Users can view their notifications" ON public.invitation_notifications
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can update their notifications" ON public.invitation_notifications
    FOR UPDATE USING (user_id = auth.uid());

-- Function to send organization invitation
CREATE OR REPLACE FUNCTION public.send_organization_invitation(
    org_id UUID,
    invite_email TEXT,
    invite_role TEXT DEFAULT 'member',
    invite_message TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    invitation_id UUID;
    invitation_token TEXT;
    target_user_id UUID;
BEGIN
    -- Check if user has permission to invite
    IF NOT EXISTS (
        SELECT 1 FROM public.organization_memberships 
        WHERE organization_id = org_id 
        AND user_id = auth.uid() 
        AND role IN ('owner', 'admin')
    ) THEN
        RAISE EXCEPTION 'Insufficient permissions to send invitations';
    END IF;
    
    -- Check if user is already a member
    SELECT id INTO target_user_id 
    FROM public.users 
    WHERE email = invite_email;
    
    IF target_user_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM public.organization_memberships 
        WHERE organization_id = org_id AND user_id = target_user_id
    ) THEN
        RAISE EXCEPTION 'User is already a member of this organization';
    END IF;
    
    -- Generate secure token
    invitation_token := encode(gen_random_bytes(32), 'base64');
    
    -- Create invitation
    INSERT INTO public.organization_invitations (
        organization_id, email, role, invited_by, message, token
    ) VALUES (
        org_id, invite_email, invite_role, auth.uid(), invite_message, invitation_token
    ) ON CONFLICT (organization_id, email) 
    DO UPDATE SET 
        role = EXCLUDED.role,
        invited_by = EXCLUDED.invited_by,
        message = EXCLUDED.message,
        token = EXCLUDED.token,
        expires_at = NOW() + INTERVAL '7 days',
        accepted_at = NULL,
        accepted_by = NULL
    RETURNING id INTO invitation_id;
    
    -- Create notification if user exists
    IF target_user_id IS NOT NULL THEN
        INSERT INTO public.invitation_notifications (user_id, invitation_id)
        VALUES (target_user_id, invitation_id)
        ON CONFLICT DO NOTHING;
    END IF;
    
    RETURN invitation_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to accept organization invitation
CREATE OR REPLACE FUNCTION public.accept_organization_invitation(
    invitation_token TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    invitation_record RECORD;
    user_email TEXT;
BEGIN
    -- Get user email
    SELECT email INTO user_email 
    FROM public.users 
    WHERE id = auth.uid();
    
    -- Get invitation details
    SELECT * INTO invitation_record
    FROM public.organization_invitations
    WHERE token = invitation_token
    AND email = user_email
    AND expires_at > NOW()
    AND accepted_at IS NULL;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Invalid or expired invitation';
    END IF;
    
    -- Check if already a member
    IF EXISTS (
        SELECT 1 FROM public.organization_memberships 
        WHERE organization_id = invitation_record.organization_id 
        AND user_id = auth.uid()
    ) THEN
        -- Mark invitation as accepted anyway
        UPDATE public.organization_invitations 
        SET accepted_at = NOW(), accepted_by = auth.uid()
        WHERE id = invitation_record.id;
        
        RETURN TRUE;
    END IF;
    
    -- Add user to organization
    INSERT INTO public.organization_memberships (
        organization_id, user_id, role, invited_by
    ) VALUES (
        invitation_record.organization_id, 
        auth.uid(), 
        invitation_record.role,
        invitation_record.invited_by
    );
    
    -- Mark invitation as accepted
    UPDATE public.organization_invitations 
    SET accepted_at = NOW(), accepted_by = auth.uid()
    WHERE id = invitation_record.id;
    
    -- Mark notification as read
    UPDATE public.invitation_notifications 
    SET is_read = TRUE
    WHERE invitation_id = invitation_record.id AND user_id = auth.uid();
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to decline organization invitation
CREATE OR REPLACE FUNCTION public.decline_organization_invitation(
    invitation_token TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    invitation_record RECORD;
    user_email TEXT;
BEGIN
    -- Get user email
    SELECT email INTO user_email 
    FROM public.users 
    WHERE id = auth.uid();
    
    -- Get invitation details
    SELECT * INTO invitation_record
    FROM public.organization_invitations
    WHERE token = invitation_token
    AND email = user_email
    AND expires_at > NOW()
    AND accepted_at IS NULL;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Invalid or expired invitation';
    END IF;
    
    -- Delete the invitation (decline)
    DELETE FROM public.organization_invitations 
    WHERE id = invitation_record.id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user's pending invitations
CREATE OR REPLACE FUNCTION public.get_user_invitations()
RETURNS TABLE (
    invitation_id UUID,
    organization_name TEXT,
    organization_description TEXT,
    role TEXT,
    invited_by_name TEXT,
    invited_by_email TEXT,
    message TEXT,
    token TEXT,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    user_email TEXT;
BEGIN
    -- Get user email
    SELECT email INTO user_email 
    FROM public.users 
    WHERE id = auth.uid();
    
    RETURN QUERY
    SELECT 
        i.id,
        o.name,
        o.description,
        i.role,
        u.email,
        u.email,
        i.message,
        i.token,
        i.expires_at,
        i.created_at
    FROM public.organization_invitations i
    JOIN public.organizations o ON i.organization_id = o.id
    JOIN public.users u ON i.invited_by = u.id
    WHERE i.email = user_email
    AND i.expires_at > NOW()
    AND i.accepted_at IS NULL
    ORDER BY i.created_at DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Update triggers for updated_at
CREATE TRIGGER update_organizations_updated_at
    BEFORE UPDATE ON public.organizations
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_equipment_updated_at
    BEFORE UPDATE ON public.equipment
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_templates_updated_at
    BEFORE UPDATE ON public.analysis_templates
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();