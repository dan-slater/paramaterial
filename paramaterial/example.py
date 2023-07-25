"""Module for functions that are used to set up the examples."""
import os
import shutil
import requests


BASE_URL = 'https://github.com/dan-slater/paramaterial-examples/raw/initial-setup/examples'
EXAMPLE_NAMES = ['example_1', 'example_2', 'example_3', 'example_4', 'example_5']


def download_example(to_directory: str, example_name: str = 'example_1'):
    """Download example data and Jupyter Notebook to the specified directory.

    Args:
        to_directory (str): The directory to download the example to.
        example_name (str): The name of the example to download.
    """

    # Check if the example name is recognized
    if example_name not in EXAMPLE_NAMES:
        raise ValueError(f'Example name {example_name} not recognized. '
                         f'Existing example names are: {", ".join(EXAMPLE_NAMES)}.')

    # Create the output directory if it doesn't exist
    os.makedirs(to_directory, exist_ok=True)

    # Download the example tarball
    url = f'{BASE_URL}/{example_name}.tar.gz'
    tarball_path = f'{example_name}.tar.gz'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(tarball_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    else:
        raise Exception(f"Download tarball error occurred: {response.status_code}")

    # Extract the example tarball
    try:
        shutil.unpack_archive(tarball_path, to_directory)
        # os.rename(extracted_dir, os.path.join(to_directory, example_name))
    except FileNotFoundError as fnf_error:
        print(f"No file: {fnf_error}")
    except Exception as err:
        print(f"An error occurred during extraction: {err}")

    # Clean up the downloaded tarball
    try:
        # pass
        os.remove(tarball_path)
    except Exception as e:
        print(f"An error occurred while deleting file : {e}")


if __name__ == '__main__':
    download_example(to_directory='examples', example_name='example_1')
