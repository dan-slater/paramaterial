""" Fitting imported constitutive models to sampled stress-strain data. [danslater, 2march2022] """
from typing import Dict

import numpy as np

import paramaterial.models as hm
from paramaterial.plug import DataItem
from paramaterial.sampling import sample

model_funcs = [
    hm.perfect,
    hm.linear,
    hm.voce,
    hm.quadratic,
    hm.ramberg
]


def pad_with_nans(data: np.ndarray, length: int):
    return np.hstack([data, np.array([None]*(length - len(data)))])


def fit_models(dataitem: DataItem, cfg: Dict):
    # get data length for padding and adding
    dataitem_length = len(dataitem.data)

    # sample data
    x_data, y_data = sample(dataitem, cfg['sampling']['sample_size'])
    dataitem.data['sampled strain'] = pad_with_nans(x_data, dataitem_length)
    dataitem.data['sampled stress'] = pad_with_nans(y_data, dataitem_length)

    # setup strain and stress vectors
    strain_vec = np.linspace(0, x_data[-1], len(x_data))  # monotonically increasing strain vector
    stress_vec = np.interp(strain_vec, x_data, y_data)  # corresponding stress vector
    dataitem.data['fitting strain'] = pad_with_nans(strain_vec, dataitem_length)
    dataitem.data['fitting stress'] = pad_with_nans(stress_vec, dataitem_length)

    # setup models
    models = []
    for model_name in cfg['models']:
        for model_func in model_funcs:
            if model_name == model_func.__name__:
                model = hm.IsoReturnMapModel(
                    name=model_name,
                    func=model_func,
                    param_names=(cfg['bounds'][model_name].keys()),
                    bounds=[eval(bounds) for bounds in cfg['bounds'][model_name].values()],
                    constraints=cfg['constraints'][model_name],
                    strain_vec=strain_vec,
                    stress_vec=stress_vec
                )
                models.append(model)

    # fit models
    print(f'Fitting models to {dataitem.test_id}.')
    models = [model.fit() for model in models]

    # save model fitting info
    for model in models:
        dataitem.info[f'{model.name} error'] = round(model.opt_res.fun, 2)  # store residual error
        for param_name, param_bounds, param_val in zip(model.param_names, model.bounds, list(model.opt_res.x)):
            dataitem.info[f'{param_name} {model.name} bounds'] = param_bounds  # store param bounds
            dataitem.info[f'{param_name} {model.name} opt val'] = round(param_val, 2)  # store optimised values

    # use models to predict stress vector, save vectors
    model_strain = np.linspace(0, x_data[-1], dataitem_length)  # artificial strain for stress prediction
    dataitem.data[f'model strain'] = model_strain
    for model in models:
        dataitem.data[f'{model.name} stress'] = model.predict(model_strain)
        dataitem.data[f'{model.name} plastic strain'] = model.predict_plastic_strain(model_strain)
        dataitem.data[f'{model.name} accumulated plastic strain'] = model.predict_accumulated_plastic_strain(model_strain)

    return dataitem
