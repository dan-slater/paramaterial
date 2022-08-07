""" Fitting imported constitutive models to sampled stress-strain data. [danslater, 2march2022] """
from typing import Dict

import numpy as np
import pandas as pd

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
    return np.hstack([data, np.array([None] * (length - len(data)))])


def fit_models(dataitem: DataItem, cfg: Dict):
    # get data length for padding and adding
    dataitem_length = len(dataitem.data)

    # sample data
    x_data, y_data = sample(dataitem, cfg['sampling']['sample_size'])  # sample data
    # x_data = np.linspace(0, x_data[-1], len(x_data))  # make strain monotonically increasing

    # x_data, y_data = dataitem.data['Strain'], dataitem.data['Stress(MPa)']

    # dataitem.data['model strain'] = pad_with_nans(strain_vec, dataitem_length)
    dataitem.data['sampled strain'] = pad_with_nans(x_data, dataitem_length)
    dataitem.data['sampled stress'] = pad_with_nans(y_data, dataitem_length)

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
                    x_data=x_data,  # actual strain data
                    # x_data=strain_vec,  # artificial strain data
                    y_data=y_data
                )
                models.append(model)

    # fit models
    model_strain = np.linspace(0, x_data[-1], 200)
    dataitem.data[f'model strain'] = pad_with_nans(model_strain, dataitem_length)

    for model in models:
        print(f'{".": <10}Fitting "{model.name}" to "{dataitem.test_id}".')
        model.fit()


        dataitem.data[f'{model.name} stress'] = pad_with_nans(model.predict(model_strain), dataitem_length)
        dataitem.data[f'{model.name} plastic strain'] = pad_with_nans(model.predict_plastic_strain(model_strain),
                                                                      dataitem_length)
        dataitem.data[f'{model.name} accumulated plastic strain'] = pad_with_nans(
            model.predict_accumulated_plastic_strain(model_strain), dataitem_length)

        dataitem.info_row[f'{model.name} error'] = round(model.opt_res.fun, 2)  # store residual error

        for param_name, param_bounds, param_val in zip(model.param_names, model.bounds, list(model.opt_res.x)):
            dataitem.info_row[f'{param_name} {model.name} bounds'] = param_bounds  # store bounds
            dataitem.info_row[f'{param_name} {model.name} opt val'] = round(param_val, 2)  # store optimised value

    return dataitem
