# ParaMaterial Migration to Supabase + Vercel

## Migration Overview

This migration transforms the Flask application from a filesystem-based approach to a modern cloud-native architecture using Supabase and Vercel.

## Key Changes

### Database & Storage
- **Before**: Local filesystem job storage in `/jobs` directory
- **After**: Supabase PostgreSQL database with structured tables and cloud storage

### Authentication
- **Before**: No user authentication
- **After**: Supabase Auth with email/password login and registration

### File Management
- **Before**: Local file uploads and manual cleanup
- **After**: Supabase Storage with automatic organization and access control

### Hosting
- **Before**: Manual Flask server deployment
- **After**: Vercel serverless deployment with automatic scaling

## New File Structure

```
webapp/
├── app_supabase.py          # New Supabase-integrated Flask app
├── config.py                # Centralized configuration
├── requirements.txt         # Updated dependencies
├── vercel.json             # Vercel deployment config
├── .env.example            # Environment template
├── supabase/
│   ├── config.toml         # Supabase project config
│   └── migrations/
│       ├── 001_initial_schema.sql
│       └── 002_storage_setup.sql
├── utils/
│   ├── supabase_client.py  # Supabase client wrapper
│   └── validation_supabase.py  # Cloud storage validation
└── templates/
    ├── login.html          # New authentication pages
    ├── register.html
    └── jobs.html           # Job management UI
```

## Database Schema

### `users` table
- Extends Supabase auth.users
- Tracks user profiles and metadata

### `jobs` table
- Job records with status tracking
- User-specific job ownership
- Error handling and metadata storage

### `job_files` table
- File metadata and storage paths
- Upload completion tracking
- File type classification

### Storage Buckets
- `job-files` bucket for secure file storage
- User-scoped access with RLS policies
- Organized by `user_id/job_id/filename`

## Migration Steps

### 1. Setup Supabase Project
```bash
# Install Supabase CLI
npm install -g supabase

# Initialize project
supabase init

# Run migrations
supabase db reset
```

### 2. Configure Environment
```bash
cp .env.example .env
# Fill in Supabase credentials
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Deploy to Vercel
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

## Benefits

### Scalability
- Automatic scaling with Vercel
- Managed database with Supabase
- Global CDN for static assets

### Security
- Row-level security policies
- Secure authentication flows
- Environment-based secrets

### Maintenance
- No server management required
- Automatic backups and updates
- Built-in monitoring and logs

### Development
- Local development with Supabase CLI
- Preview deployments for testing
- Git-based deployment workflow

## Breaking Changes

1. **User Authentication Required**: All job operations now require user login
2. **API Changes**: Routes updated to use database instead of filesystem
3. **File Access**: Files now accessed through Supabase Storage URLs
4. **Configuration**: Environment variables required for Supabase connection

## Testing

### Local Development
```bash
# Start Supabase locally
supabase start

# Run Flask app
python app_supabase.py
```

### Production Testing
- Use Vercel preview deployments
- Test with production Supabase instance
- Verify authentication flows

## Rollback Plan

If migration issues occur:
1. Original `app.py` and `validation.py` are preserved
2. Switch back to filesystem-based approach
3. Restore local job directories if needed
4. Use original deployment method