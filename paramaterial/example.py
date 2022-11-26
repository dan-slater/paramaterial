"""Module for functions that are used to set up the examples."""


def download_example_data():
    """Download example data from the internet."""
    import urllib.request
    import zipfile
    import os
    import pandas as pd
    import paramaterial as pam
    import numpy as np

    # make data and info directories if they don't exist
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('info'):
        os.mkdir('info')

    print('Downloading example data...')
    
    # Download the data
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/rd6jm9tyb6-2.zip'
    urllib.request.urlretrieve(url, 'example_data.zip')

    # Unzip the data and store in a folder called data/00 raw data
    with zipfile.ZipFile('example_data.zip', 'r') as zip_ref:
        zip_ref.extractall('data/00 raw data')

    # if data/00 raw data contains a zip file, unzip it
    for file in os.listdir('data/00 raw data'):
        if file.endswith('.zip'):
            with zipfile.ZipFile('data/00 raw data/' + file, 'r') as zip_ref:
                zip_ref.extractall('data/00 raw data')

    # any zip files in data/00 raw data
    for file in os.listdir('data/00 raw data'):
        if file.endswith('.zip'):
            os.remove('data/00 raw data/' + file)

    os.remove('example_data.zip')

    # Print a message
    print(f'Example data downloaded, unzipped, and saved to {os.path.join(os.getcwd(), "data/00 raw data")}.')

    # Make the info table
    info_lists = [[filename] + filename.split('_')[:4] for filename in os.listdir('data/00 raw data')]
    info_table = pd.DataFrame(info_lists,
                              columns=['old filename', 'test type', 'temperature', 'lot', 'number']
                              ).sort_values(by='test type', ascending=False)
    info_table['test id'] = [f'test_ID_{i + 1:03d}' for i in range(len(info_table))]
    info_table = info_table.set_index('test id').reset_index()
    info_table['test type'] = info_table['test type'].replace('T', 'UT')
    info_table['test type'] = info_table['test type'].replace('P', 'PST')
    info_table['rate'] = 8.66e-4  # units (/s) and all tests performed at same rate
    info_table['A_0'] = np.where(info_table['test type'] == 'UT', 40.32, 20.16)
    info_table['h_0'] = 3.175
    info_table['temperature (C)'] = pd.to_numeric(info_table['temperature'])

    # Check raw data
    pam.preparing.check_column_headers('data/00 raw data')
    pam.preparing.check_for_duplicate_files('data/00 raw data')
    
    # Make prepared data
    pam.preparing.copy_data_and_rename_by_test_id('data/00 raw data', 'data/01 prepared data', info_table)
    
    # Save the info table
    info_table.to_csv('info/01 prepared info.csv', index=False)
    
    # Print a message
    print(f'Info table saved to {os.path.join(os.getcwd(), "info/01 prepared info.csv")}.')
    

if __name__ == '__main__':
    # Download the example data
    download_example_data()
