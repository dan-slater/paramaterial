# ParaMaterial Database Schema Roadmap

## Current Schema Overview

### Authentication & Users (Migration 001)
```sql
public.users
├── id UUID (references auth.users)
├── email TEXT
├── created_at TIMESTAMP
└── updated_at TIMESTAMP
```

### Core Job Management (Migration 001)
```sql
public.jobs
├── id UUID (primary key)
├── user_id UUID (references users)
├── status TEXT (pending, uploading, validating, processing, completed, failed)
├── created_at, updated_at, completed_at TIMESTAMP
├── error_message TEXT
└── metadata JSONB

public.job_files
├── id UUID (primary key)  
├── job_id UUID (references jobs)
├── file_name TEXT
├── file_type TEXT (info_table, time_series)
├── file_size INTEGER
├── storage_path TEXT (Supabase Storage path)
├── upload_completed BOOLEAN
└── created_at TIMESTAMP
```

### Storage & Security (Migration 002)
- Storage bucket 'job-files' with RLS policies
- User-scoped file access: user_id/job_id/filename
- File type restrictions: CSV, XLSX only
- 50MB size limit

### Organizations (Migration 003)
```sql
public.organizations
├── id UUID
├── name TEXT
├── description TEXT  
├── logo_url TEXT
├── settings JSONB
└── timestamps

public.organization_memberships  
├── organization_id, user_id (composite key)
├── role TEXT (owner, admin, member, viewer)
├── joined_at TIMESTAMP
└── invited_by UUID

public.organization_invitations
├── organization_id, email
├── role TEXT, invited_by UUID
├── token TEXT (secure random)
├── expires_at TIMESTAMP (7 days)
├── accepted_at, accepted_by
└── message TEXT

public.equipment
├── organization_id UUID
├── name, model, description TEXT
├── equipment_type TEXT (gleeble, sem, ebsd)
├── specifications JSONB
├── location TEXT
└── is_active BOOLEAN

public.analysis_templates
├── organization_id, equipment_id UUID
├── created_by UUID
├── name, description TEXT
├── template_type TEXT (processing, analysis, visualization)
├── parameters JSONB
├── is_public BOOLEAN
├── usage_count INTEGER
└── tags TEXT[]
```

## Planned Schema Updates

### Phase 1: Essential Features

#### Update 1: Enhanced Job Context
```sql
-- Add to existing jobs table
ALTER TABLE public.jobs ADD COLUMN IF NOT EXISTS:
├── template_id UUID REFERENCES analysis_templates(id)
├── equipment_id UUID REFERENCES equipment(id)  
├── project_id UUID REFERENCES research_projects(id)
├── template_version INTEGER DEFAULT 1
└── processing_parameters JSONB -- actual parameters used
```

#### Update 2: Template Categories & Organization
```sql
-- Add to analysis_templates table
ALTER TABLE public.analysis_templates ADD COLUMN IF NOT EXISTS:
├── category TEXT -- 'heat_treatment', 'microstructure', 'mechanical_testing'
├── version INTEGER DEFAULT 1
├── parent_template_id UUID REFERENCES analysis_templates(id)
├── is_latest_version BOOLEAN DEFAULT TRUE
└── changelog TEXT

-- Template categories table
CREATE TABLE public.template_categories (
    id UUID PRIMARY KEY,
    organization_id UUID REFERENCES organizations(id),
    name TEXT NOT NULL,
    description TEXT,
    color TEXT, -- hex color for UI
    icon TEXT,  -- icon name
    sort_order INTEGER DEFAULT 0
);
```

#### Update 3: User Preferences & Settings
```sql
CREATE TABLE public.user_preferences (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    default_organization_id UUID REFERENCES organizations(id),
    ui_theme TEXT DEFAULT 'plasma',
    notification_settings JSONB DEFAULT '{"email": true, "in_app": true}',
    preferred_units JSONB DEFAULT '{"temperature": "celsius", "pressure": "mpa"}',
    dashboard_layout JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Phase 2: Enhanced Features

#### Update 4: Research Project Management
```sql
-- New table for organizing related jobs
CREATE TABLE public.research_projects (
    id UUID PRIMARY KEY,
    organization_id UUID REFERENCES organizations(id),
    created_by UUID REFERENCES users(id),
    name TEXT NOT NULL,
    description TEXT,
    project_code TEXT, -- e.g., "CME-2024-001"
    status TEXT CHECK (status IN ('active', 'completed', 'archived')),
    start_date DATE,
    end_date DATE,
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Update 5: Template Versioning System
```sql
-- Template version history
CREATE TABLE public.template_versions (
    id UUID PRIMARY KEY,
    template_id UUID REFERENCES analysis_templates(id),
    version_number INTEGER NOT NULL,
    parameters JSONB NOT NULL,
    changelog TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(template_id, version_number)
);
```

### Phase 3: Advanced Features

#### Update 6: Activity & Audit Logging
```sql
CREATE TABLE public.activity_log (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),
    action_type TEXT NOT NULL, -- 'job_created', 'template_shared', 'invitation_sent'
    resource_type TEXT, -- 'job', 'template', 'organization'
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Implementation Priority

### Phase 1 (Essential) - Implement First
1. **Enhanced Job Context** - Link jobs to templates and equipment
2. **Template Categories** - Organize templates by research type  
3. **User Preferences** - Basic user settings

### Phase 2 (Valuable) - Implement After Core Testing
4. **Research Projects** - Group related jobs together
5. **Template Versioning** - Track template evolution

### Phase 3 (Advanced) - Future Enhancements
6. **Activity Logging** - Audit trail for compliance

## Excluded Features
- Equipment scheduling and reservations (not needed for UCT use case)
- Equipment maintenance tracking (handled externally)

## UCT Centre for Materials Engineering Use Cases

### Equipment Types
- Gleeble 3800 (Thermomechanical Testing)
- JEOL JSM-7001F SEM
- EBSD System
- Universal Testing Machines

### Template Categories
- Heat Treatment Analysis
- Microstructure Characterization
- Mechanical Property Testing
- Failure Analysis
- Additive Manufacturing
- Coating Analysis

### Research Areas
- Light Metals Research (Ti-6Al-4V, Aluminum alloys)
- Additive Manufacturing (Laser powder bed fusion)
- Thermal Processing
- Materials Characterization

## Benefits
1. **Consistency**: Standardized analysis across student projects
2. **Knowledge Transfer**: Senior students share proven configurations
3. **Quality Control**: Validated templates ensure reliable results
4. **Efficiency**: New students start with working configurations
5. **Collaboration**: Template sharing within research teams
6. **Documentation**: Automatic tracking of analysis parameters
7. **Reproducibility**: Exact parameter replication for validation