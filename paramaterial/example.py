"""
The example module is designed to facilitate the downloading and setup of predefined examples for the Paramaterial
library.

Function:
    - `download_example`: A function to download a specified example, extract its content, and save it to a
    designated directory.

Currently Available Examples:
    - 'dan_msc_basic_usage_0.1.0'
    - 'dan_msc_cs1_0.1.0'
    - 'dan_msc_cs2_0.1.0'
    - 'dan_msc_cs3_0.1.0'
    - 'dan_msc_cs4_0.1.0'

Examples:
    >>> # Download the 'dan_msc_basic_usage_0.1.0' example to the current directory
    >>> from paramaterial import download_example
    >>> download_example('dan_msc_basic_usage_0.1.0')

About the Example Repository:
    The examples are hosted in a GitHub repository and include datasets, notebooks, and other assets that showcase
    the functionality and capabilities of the Paramaterial library. These examples can be downloaded and run locally,
    providing an interactive way to explore and learn about the library.

    The `download_example` function allows users to fetch any of the available examples by name. It takes care of
    downloading and extracting the data, info, and notebook files to the specified directory.
"""
import os
import shutil
import requests

BASE_URL = 'https://github.com/dan-slater/paramaterial-examples/raw/main/examples'
EXAMPLE_NAMES = ['dan_msc_basic_usage_0.1.0', 'dan_msc_cs1_0.1.0', 'dan_msc_cs2_0.1.0',
                 'dan_msc_cs3_0.1.0', 'dan_msc_cs4_0.1.0']


def download_example(example_name: str, to_directory: str = './'):
    """Download and extract an example from the Paramaterial example repository.

    Args:
        example_name: The name of the example to download. Must be one of the predefined
            examples available in the Paramaterial example repository. Examples include:

            - 'dan_msc_basic_usage_0.1.0'
            - 'dan_msc_cs1_0.1.0'
            - 'dan_msc_cs2_0.1.0'
            - 'dan_msc_cs3_0.1.0'
            - 'dan_msc_cs4_0.1.0'
        to_directory: The directory to download and extract the example to.

    Raises:
        ValueError: If the specified example_name is not recognized.
        Exception: If an error occurs during the download or extraction process.

    Examples:
        >>> download_example('dan_msc_basic_usage_0.1.0')
        Example dan_msc_basic_usage_0.1.0 downloaded to ./
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

    print(f"Example {example_name} downloaded to {to_directory}")
