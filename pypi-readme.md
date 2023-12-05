![Paramaterial logo](https://github.com/dan-slater/paramaterial/blob/main/docs/img/paramaterial-logo.png?raw=true)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paramaterial)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/paramaterial)
![Libraries.io dependency status for latest release](https://img.shields.io/librariesio/release/pypi/paramaterial)
[![PyPI version](https://badge.fury.io/py/paramaterial.svg)](https://badge.fury.io/py/paramaterial)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/dan-slater/paramaterial?include_prereleases)

[//]: # (![PyPI - Downloads]&#40;https://img.shields.io/pypi/dm/paramaterial&#41;)
[//]: # (![GitHub search hit counter]&#40;https://img.shields.io/github/search/dan-slater/paramaterial/goto&#41;)

## About

A Python package for parameterizing materials test data. Given a set of experimental measurements, Paramaterial can be
used to determine material properties and constitutive model parameters.

* **Source Code**: [https://github.com/dan-slater/paramaterial](https://github.com/dan-slater/paramaterial)

* **Documentation**: [https://dan-slater.github.io/paramaterial/](https://dan-slater.github.io/paramaterial/)

* **PyPI**: [https://pypi.org/project/paramaterial/](https://pypi.org/project/paramaterial/)

* **Examples**: [https://github.com/dan-slater/paramaterial-examples](https://github.com/dan-slater/paramaterial-examples)

The package was designed to help improve the quality and quantity of data available for materials modeling and
simulation. It is hoped that using Paramaterial will help improve repeatability and reproducibility of materials test
data analyses, and help to reduce the time and effort required to perform such analyses.


## Installation

```shell
pip install paramaterial
```

## Usage

Please see the [API reference](reference/example.md) for details on the toolkit's
functions and classes.

Usage examples are available in the [examples repository](https://github.com/dan-slater/paramaterial-examples).
These examples can be downloaded using the `download_example` function:

```python
# Download the basic usage example to the current directory
from paramaterial import download_example
download_example('dan_msc_basic_usage_0.1.0')
# Other examples are also currently available:
# download_example('dan_msc_cs1_0.1.0')
# download_example('dan_msc_cs2_0.1.0')
# download_example('dan_msc_cs3_0.1.0')
# download_example('dan_msc_cs4_0.1.0')
```

The examples include datasets, notebooks, and other assets that showcase the functionality and capabilities of the Paramaterial library. These examples can be downloaded and run locally, providing an interactive way to explore and learn about the library.
For more details see the documentation for the `download_example` function at [reference/example](reference/example.md).
## Overview

Paramaterial is an open-source Python package for parameterising materials test
data. Paramaterial provides functionality for the repeatable processing of mechanical test results,
such as stress-strain data from a tensile test. An example of various stages of data processing that
might be performed using the toolkit is shown in the figure below.

![Paramaterial overview](https://github.com/dan-slater/paramaterial/blob/main/docs/img/readme-graphic-1.png?raw=true)

Paramaterial is also useful for generating a table of parameters from raw data, as illustrated below. Various data analysis techniques can then be applied to this table of parameters.

![Paramaterial overview](https://github.com/dan-slater/paramaterial/blob/main/docs/img/readme-graphic-2.png?raw=true)

## Contributing

Please go to the GitHub repository and submit an issue or pull request.

## License

Paramaterial is licensed under the MIT license. 


