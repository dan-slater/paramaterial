import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def validate_series_files(series_dir, info_ids):
    errors = []
    found_files = []
    # Check only files directly in series_dir, ignore subdirectories
    for item_name in os.listdir(series_dir):
        item_path = os.path.join(series_dir, item_name)
        if os.path.isfile(item_path) and (item_name.endswith('.csv') or item_name.endswith('.xlsx')):
            base = os.path.splitext(item_name)[0]
            found_files.append(base)
            # Optional: Add check for empty files here if needed
            # try:
            #     if os.path.getsize(item_path) == 0:
            #         errors.append(f"Series file is empty: {item_name}")
            # except OSError as e:
            #     errors.append(f"Could not check size of file {item_name}: {e}")

    # Convert both to sets for efficient comparison
    info_ids_set = set(map(str, info_ids)) # Ensure info_ids are strings
    found_files_set = set(found_files)

    missing_series = info_ids_set - found_files_set
    if missing_series:
        errors.append(f"Missing series file(s) for ID(s): {', '.join(sorted(list(missing_series)))}")

    # Optional: Check for extra files not listed in the info table
    extra_files = found_files_set - info_ids_set
    if extra_files:
        errors.append(f"Found extra series file(s) not listed in info table: {', '.join(sorted(list(extra_files)))}")

    return len(errors) == 0, errors


def extract_ids_from_info_table(info_path):
    """Reads an info table (CSV or XLSX) and extracts IDs.

    Args:
        info_path (str): The path to the info table file.

    Returns:
        list: A list of IDs found in the 'ID' or 'test_id' column, converted to strings.

    Raises:
        ValueError: If the file cannot be read, is empty, is not a CSV/XLSX,
                    or does not contain an 'ID' or 'test_id' column.
    """
    logger.debug(f"Attempting to read info table: {info_path}")
    try:
        if info_path.endswith('.csv'):
            df = pd.read_csv(info_path)
        elif info_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(info_path)
        else:
            raise ValueError(f"Unsupported file type: {info_path}. Please upload CSV or XLSX.")

        if df.empty:
            raise ValueError(f"Info table is empty: {info_path}")

        id_column = None
        if 'ID' in df.columns:
            id_column = 'ID'
        elif 'test_id' in df.columns:
            id_column = 'test_id'
        else:
            raise ValueError("Info table must contain an 'ID' or 'test_id' column.")

        logger.debug(f"Found ID column: '{id_column}'")
        # Ensure IDs are strings to match file naming conventions
        ids = df[id_column].astype(str).tolist()
        logger.debug(f"Extracted {len(ids)} IDs from info table.")
        return ids

    except FileNotFoundError:
        logger.error(f"Info table file not found at path: {info_path}")
        raise ValueError(f"Info table file not found: {os.path.basename(info_path)}")
    except pd.errors.EmptyDataError:
        # This might be redundant due to the df.empty check, but good practice
        logger.error(f"Pandas reported empty data error for: {info_path}")
        raise ValueError(f"Info table appears to be empty or corrupted: {os.path.basename(info_path)}")
    except Exception as e:
        logger.error(f"Failed to read or process info table '{info_path}': {e}", exc_info=True)
        # Re-raise as ValueError or a more specific custom exception if needed
        raise ValueError(f"Error processing info table: {e}")
