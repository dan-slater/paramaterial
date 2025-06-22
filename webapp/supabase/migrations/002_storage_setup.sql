-- Create storage bucket for job files
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'job-files',
    'job-files',
    false,
    52428800, -- 50MB in bytes
    ARRAY['text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']
);

-- Create storage policies for job files bucket
CREATE POLICY "Users can upload files to own jobs" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'job-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "Users can view files from own jobs" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'job-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "Users can update files from own jobs" ON storage.objects
    FOR UPDATE USING (
        bucket_id = 'job-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "Users can delete files from own jobs" ON storage.objects
    FOR DELETE USING (
        bucket_id = 'job-files' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- Create function to generate storage path for job files
CREATE OR REPLACE FUNCTION public.generate_storage_path(user_id UUID, job_id UUID, filename TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN user_id::text || '/' || job_id::text || '/' || filename;
END;
$$ LANGUAGE plpgsql;