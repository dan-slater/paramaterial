"""Fit the Ramberg-Osgood model to the representative curves in modelling_test_data and compare the fitted parameters
with tests/reference/modelling.json, which was recorded with paramaterial 0.1.3 (pandas 1.5.3, numpy 1.24.4,
scipy 1.10.1). These are the curves and the fit from the MSc thesis basic-usage example.

Re-record the reference (only on purpose) with:

    PYTHONPATH=<paramaterial-0.1.3 source> python tests/test_modelling.py --write-reference
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import paramaterial as pam

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'modelling_test_data'
REFERENCE = HERE / 'reference' / 'modelling.json'

# The minimum the optimiser finds (the fit error) matches the reference to ~1e-10 on every scipy tried (1.10 to 1.18,
# numpy 1.24 to 2.5). Where L-BFGS-B stops along the flat H-n valley moves the 6th significant figure of H and n
# (worst seen 9.4e-6), so those, and the curves predicted from them, are held to 1e-4. The thesis reports them to
# 3 or 4 significant figures.
EXACT_RTOL = 1e-12
ERROR_RTOL = 1e-9
PARAM_RTOL = 1e-4


def fit_ramberg() -> dict:
    ds = pam.DataSet(str(DATA_DIR / '03 temp repres info.xlsx'), str(DATA_DIR / 'repres_data'),
                     test_id_key='repres_id')
    ms = pam.ModelSet(pam.models.ramberg, var_names=['E', 'UPL_1'], param_names=['H', 'n'],
                      bounds=[(0., 1000.), (0.01, 0.8)])
    ms.fit_to(ds, x_key='Strain', y_key='Stress_MPa')
    prediction = ms.predict(x_range=(0, 0.1, 0.001))
    results = ms.fitting_results
    return {
        'columns': list(results.columns),
        'rows': [[str(row[0])] + [float(v) for v in row[1:]] for row in results.itertuples(index=False)],
        'predicted_stress_sum': {di.test_id: float(di.data['Stress_MPa'].sum()) for di in prediction},
    }


@pytest.fixture(scope='module')
def fitted():
    return fit_ramberg()


@pytest.fixture(scope='module')
def reference():
    return json.loads(REFERENCE.read_text())


def test_fitting_table_shape(fitted, reference):
    assert fitted['columns'] == reference['columns'] == ['model_id', 'E', 'UPL_1', 'H', 'n', 'error']
    assert [r[0] for r in fitted['rows']] == [r[0] for r in reference['rows']]


def test_fitted_parameters_match_reference(fitted, reference):
    new, ref = np.array([r[1:] for r in fitted['rows']]), np.array([r[1:] for r in reference['rows']])
    np.testing.assert_allclose(new[:, 0:2], ref[:, 0:2], rtol=EXACT_RTOL, err_msg='E, UPL_1 (inputs)')
    np.testing.assert_allclose(new[:, 2:4], ref[:, 2:4], rtol=PARAM_RTOL, err_msg='H, n')
    np.testing.assert_allclose(new[:, 4], ref[:, 4], rtol=ERROR_RTOL, err_msg='error')


def test_predictions_match_reference(fitted, reference):
    assert fitted['predicted_stress_sum'].keys() == reference['predicted_stress_sum'].keys()
    np.testing.assert_allclose(list(fitted['predicted_stress_sum'].values()),
                               list(reference['predicted_stress_sum'].values()), rtol=PARAM_RTOL)


if __name__ == '__main__':
    if '--write-reference' in sys.argv:
        REFERENCE.parent.mkdir(exist_ok=True)
        REFERENCE.write_text(json.dumps(fit_ramberg(), indent=1) + '\n')
        print(f'wrote {REFERENCE} using paramaterial {pam.__version__} from {pam.__file__}')
