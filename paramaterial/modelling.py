from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as op

from paramaterial.plug import DataSet, DataItem


class Model:
    def __init__(self,
                 objective_function: Callable[[List[float], DataItem], float],
                 initial_guess: List[float],
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 dataitem_func: Optional[Callable[[pd.Series, List[float]], DataItem]] = None,
                 opt_result: Optional[op.OptimizeResult] = None):
        self.objective_function = objective_function
        self.initial_guess = np.array(initial_guess)
        self.bounds = bounds
        self.dataitem_func = dataitem_func
        self.opt_result = opt_result
        self.fitted_ds: DataSet | None = None

    def fit_item(self, di: DataItem) -> DataItem:
        def di_obj_func(params) -> float:
            return self.objective_function(params, di)

        self.opt_result = op.minimize(di_obj_func, self.initial_guess, bounds=self.bounds)
        test_id = di.test_id

        return self.predict_item(di.info)

    def fit_to(self, ds: DataSet, **kwargs) -> None:
        def ds_obj_func(params) -> float:
            return sum([self.objective_function(params, di) for di in ds])

        self.opt_result = op.minimize(ds_obj_func, self.initial_guess, bounds=self.bounds, **kwargs)
        self.fitted_ds = ds

    def predict_item(self, info: pd.Series) -> DataItem:
        return self.dataitem_func(info, self.opt_result.x)

    def predict_set(self, info_table: Optional[pd.DataFrame]) -> DataSet:
        new_set = DataSet('model', 'model')
        # dataset.info_table = info_table
        #     dataset.data_map = map(self.predict_item, [info for _, info in info_table.iterrows()])
        #     return dataset
        # elif dataset is not None:
        #     dataset.data_map = map(self.predict_item, [info for _, info in info_table.iterrows()])
        # else:
        #     new_set.info_table = self.fitted_ds.info_table
        #     new_set.data_map = map(self.predict_item, [info for _, info in self.fitted_ds.info_table.iterrows()])




def sample(dataitem: DataItem, sample_size: int, delete_neg_strain: bool = True):
    dataitem.info['nr of points sampled'] = sample_size
    df = dataitem.data

    x_data = df['Strain'].values
    y_data = df['Stress(MPa)'].values

    if delete_neg_strain:
        for i, x_val in enumerate(x_data):
            if x_val < 0:
                x_data = np.delete(x_data, [i])
                y_data = np.delete(y_data, [i])

    sampling_stride = int(len(x_data) / sample_size)
    if sampling_stride < 1:
        sampling_stride = 1

    x_data = x_data[::sampling_stride]
    y_data = y_data[::sampling_stride]

    return x_data, y_data

# todo: sample data with minimum in given area (i.e both x and y tolerance)
# todo: variable sampling runtime paramaterial
# todo: full elastic range with plastic sampling only
# todo: equidistant strain sampling
# todo: if strain decreasing, omit point
