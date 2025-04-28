import os
import io
import tempfile
import pytest
import pandas as pd
from webapp.app import app

def test_index_route():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'ParaMaterial Web App' in response.data

def test_upload_invalid_file(monkeypatch):
    client = app.test_client()
    data = {
        'file': (io.BytesIO(b'invalid content'), 'test.txt')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert b'Invalid file type' in response.data or b'No selected file' in response.data

def test_upload_valid_csv(tmp_path):
    client = app.test_client()
    # Create a valid CSV with Strain and Stress columns
    csv_content = b'Strain,Stress\n0.01,10\n0.02,20'
    data = {
        'file': (io.BytesIO(csv_content), 'test.csv')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert b'Processing Results' in response.data or b'File validated and parsed' in response.data

def test_upload_missing_columns(tmp_path):
    client = app.test_client()
    # CSV missing required columns
    csv_content = b'Time,Force\n1,100\n2,200'
    data = {
        'file': (io.BytesIO(csv_content), 'test.csv')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert b'File must contain columns for Strain and Stress' in response.data

def test_upload_large_file(monkeypatch):
    client = app.test_client()
    # Simulate a large file by monkeypatching os.path.getsize
    csv_content = b'Strain,Stress\n0.01,10\n0.02,20'
    data = {
        'file': (io.BytesIO(csv_content), 'test.csv')
    }
    orig_getsize = os.path.getsize
    monkeypatch.setattr(os.path, 'getsize', lambda path: 11 * 1024 * 1024)  # >10MB
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    monkeypatch.setattr(os.path, 'getsize', orig_getsize)
    assert b'File too large' in response.data
