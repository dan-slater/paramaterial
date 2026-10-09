![Paramaterial logo](https://github.com/dan-slater/paramaterial/blob/main/docs/img/paramaterial-logo.png?raw=true)

[![PyPI version](https://img.shields.io/pypi/v/paramaterial)](https://pypi.org/project/paramaterial/)
![Python versions](https://img.shields.io/pypi/pyversions/paramaterial)
[![Tests](https://github.com/dan-slater/paramaterial/actions/workflows/test.yml/badge.svg)](https://github.com/dan-slater/paramaterial/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dan-slater/paramaterial/blob/main/LICENSE)

Paramaterial batch-processes mechanical test data: point it at a folder of stress-strain CSV files and a table
describing each test, and it finds properties such as elastic modulus, proportional limits, proof stress and UTS for
every test. It can then screen, average and plot the results, and fit constitutive models such as Ramberg-Osgood to
them, so the whole analysis is a script you can rerun.

## Install

```shell
pip install paramaterial
```

Python 3.9 or later.

## Example

Given a folder `tensile_tests/` with one CSV per test, each with `Strain` (mm/mm) and `Stress_MPa` columns:

```python
from pathlib import Path
import pandas as pd
import paramaterial as pam

# one CSV per test, each with Strain (mm/mm) and Stress_MPa columns
tests = sorted(Path('tensile_tests').glob('*.csv'))
pd.DataFrame({'test_id': [f.stem for f in tests]}).to_csv('info.csv', index=False)
ds = pam.DataSet('info.csv', 'tensile_tests')

ds = pam.find_UTS(ds)  # ultimate tensile strength: UTS_1 (MPa) at strain UTS_0

def elastic_region(di):  # the modulus and proof stress are found below 1% strain
    di.data = di.data[di.data['Strain'] < 0.01]
    return di

ds = pam.find_upl_and_lpl(ds.apply(elastic_region), preload=36)  # proportional limits and modulus E
ds = pam.find_proof_stress(pam.correct_foot(ds), proof_strain=0.002)  # 0.2% proof stress PS_0.002_1
print(ds.info_table[['test_id', 'E', 'PS_0.002_1', 'UTS_1']])
ds.write_output('processed_info.csv', 'processed_data')
```

`info.csv` is where test metadata goes. Add columns such as temperature, material or batch, and use them to group,
filter, colour plots and build representative curves. `preload=36` ignores points below 36 MPa when fitting the
elastic line; set it for your machine. Different column names are passed with `strain_key=` and `stress_key=`.

## Documentation and examples

- Documentation, how-to guides and API reference: https://dan-slater.github.io/paramaterial/
- Example notebooks with data: https://github.com/dan-slater/paramaterial-examples. These are the basic-usage
  walkthrough and the four case studies from the thesis: tensile tests across lots and temperatures, uniaxial versus
  plane-strain tension, and uniaxial and plane-strain compression with Zener-Hollomon fits. Download one with `pam.download_example('dan_msc_basic_usage_0.1.0')`.
- Changes between versions: [CHANGELOG.md](https://github.com/dan-slater/paramaterial/blob/main/CHANGELOG.md)
- Web lab: coming.

## Citing

Paramaterial was developed for an MSc at the University of Cape Town:

> Slater, D. (2023). *Development of Open-Source Software for Repeatable Processing of Mechanical Test Data.*
> MSc thesis, University of Cape Town. http://hdl.handle.net/11427/39860

[CITATION.cff](https://github.com/dan-slater/paramaterial/blob/main/CITATION.cff) has the details. GitHub's "Cite this repository" button reads it.

## Contributing

Issues and pull requests are welcome at https://github.com/dan-slater/paramaterial. To run the tests:

```shell
pip install -e ".[test]"
pytest
```

[tests/reference](https://github.com/dan-slater/paramaterial/blob/main/tests/reference/README.md) explains how the thesis example results are checked across dependency versions.

## License

MIT
