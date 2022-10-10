![Paramaterial logo](docs/img/paramaterial-logo.png)
[![PyPI version](https://badge.fury.io/py/paramaterial.svg)](https://badge.fury.io/py/paramaterial)

# Paramaterial
Post-processing toolkit for mechanical testing data. 

* **Source Code**: [https://github.com/dan-slater/paramaterial](https://github.com/dan-slater/paramaterial)

* **Documentation**: [https://dan-slater.github.io/paramaterial/](https://dan-slater.github.io/paramaterial/)

Use Paramaterial to:
- Process large datasets of test measurements to automatically identify material parameters and fit consitutive models.
- Produce publication-quality plots and test reports to document raw data, processed data and analysis.
- Improve efficiency and traceability of processing to increase data quality.

## Installation

> **Note**: Paramaterial is currently in alpha development. The API is subject to change.

## Usage

Usage of the toolkit is demonstrated in the [examples](

## Contributing

## License



Have you ever had to process a bunch of csv output from a mechanical test machine, copying and pasting data into a hacky Excel template to calculate things like elastic modulus and yield strength?

Only to then have to make another Excel file where you create a summary table...

And then have to copy and paste that into a report or an email...

And then you have to plot the data in Excel and spend half an hour tweaking the colours to get it to look at least halfway professional...

And then you discover Excel has formatted your strain column as a date for literally no reason so now your plots have broken...

And then next week you have to do all this again! :angry:

**No more!** :boom:

pymechtest has a very simple goal: to reduce the amount of time engineers spend munging data after a batch of mechanical testing.

Here is a quick taste of how easy it is to go from raw data to a gorgeous stress-strain plot:

The key features are:

* **Versatile**: The code design is sufficiently general so that it should work for a wide range of material tests.
* **Easy-to-use**

Also handle data in bulk useful. Improve efficiency, traceability and repeatability.

Automated generation of test reports.

* **Transparent**: Using open-source code with version markers for post-processing of test measurements provides transparency.
* **Sensible Defaults**: The API is designed around sensible defaults for things like modulus strain range, whether to expect a yield strength etc.
* **Automatic Calculations**: pymechtest will automatically calculate strength, elastic modulus, yield strength etc. for you.
* **Elegant Looking Stress Strain Curves**: pymechtest uses [altair] to plot amazing looking stress strain curves.
* **Reliable**: pymechtest uses battle-tested libraries like [pandas], [numpy] and [altair] to do most of the work. The API is really a domain-specific convenience wrapper. pymechtest also maintains high test coverage.




## Installation

```shell
pip install paramaterial
```



