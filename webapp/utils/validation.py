import pandas as pd
import os

def validate_info_table(info_path):
    try:
        if info_path.endswith('.csv'):
            df = pd.read_csv(info_path)
        elif info_path.endswith('.xlsx'):
            df = pd.read_excel(info_path)
        else:
            return False, ['Unsupported file type']
        if df.shape[1] < 2:
            return False, ['Info table must have at least two columns (ID + variable(s))']
        if df.shape[0] < 1:
            return False, ['Info table must have at least one row']
        if df.columns[0].lower() not in ['test id', 'id', 'sample id']:
            return False, ['First column must be an identifier (e.g., "Test ID")']
        return True, []
    except Exception as e:
        return False, [str(e)]

def validate_series_files(series_dir, info_ids):
    errors = []
    found_files = []
    for fname in os.listdir(series_dir):
        if fname.endswith('.csv') or fname.endswith('.xlsx'):
            base = os.path.splitext(fname)[0]
            found_files.append(base)
    missing = set(info_ids) - set(found_files)
    if missing:
        errors.append(f"Missing series files for: {', '.join(missing)}")
    return len(errors) == 0, errors
