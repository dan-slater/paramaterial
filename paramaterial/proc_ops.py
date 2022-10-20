"""Functions for post-processing material test data. (Stress-strain)"""
import copy
import os
from functools import wraps
from itertools import combinations, permutations

from typing import Callable, Dict, Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
# from mpl_interactions import zoom_factory, panhandler
# from mpl_point_clicker import clicker

from paramaterial.plug import DataSet, DataItem


# def process_data(dataitem: DataItem, cfg: Dict):
#     """ Apply processing functions to a datafile object. """
#     processing_operations = [calculate_force_disp_from_eng_curve, trim_using_max_force,
#                              calculate_eng_stress_strain_gradient, calculate_elastic_modulus,
#                              calculate_offset_yield_point,
#                              # select_pois_manually,
#                              ]
#     dataitem = store_initial_indices(dataitem)
#     for proc_op in processing_operations:  # todo: change order of error check
#         if proc_op.__name__ in cfg['operations']:
#             print(f'{".": <10}Running {proc_op.__name__}.')
#             dataitem = proc_op(dataitem)
#         else:
#             print('!! Processing operation selection error. !!')
#     return dataitem
#
#
# def processing_function(function: Callable[[DataItem, ...], DataItem]):
#     """ Applies function to dataitem then returns it. Just returns dataitem if any exception raised. """
#
#     @wraps(function)
#     def wrapper(dataitem: DataItem, *args, **kwargs):
#         input_dataitem = copy.deepcopy(dataitem)
#         try:
#             dataitem = function(dataitem, *args, **kwargs)
#         except Exception as e:
#             print(f'!! Processing function error. !!')
#             print(f'!! {function.__name__} failed for {dataitem.test_id}. !!')
#             print(e)
#         try:
#             assert input_dataitem in dataitem
#         except AssertionError as e:
#             print(f'!! Processing function error. !!')
#             print(f'!! {function.__name__} failed for {dataitem.test_id}. !!')
#             print('!! Input data and info was not in output DataItem. !!')
#             print(e)
#             dataitem = input_dataitem
#         return dataitem
#
#     return wrapper
#
#
# @processing_function
# def store_initial_indices(dataitem):
#     df = dataitem.data
#     dataitem.info['raw data indices'] = (0, len(df))
#     dataitem.data = df[:].reset_index(drop=False)
#     return dataitem
#
#
# @processing_function
# def calculate_force_disp_from_eng_curve(dataitem: DataItem) -> DataItem:
#     e = dataitem.data['eng strain']
#     s = dataitem.data['eng stress']
#     L_0 = dataitem.info['L_0 (mm)']
#     A_0 = dataitem.info['A_0 (mm^2)']
#     dataitem.data['Jaw(mm)'] = e.values*L_0
#     dataitem.data['Force(kN)'] = s.values*A_0*0.001
#     return dataitem
#
#
# @processing_function
# def trim_using_max_force(dataitem):
#     df = dataitem.data
#     maxdex = df['Force(kN)'].idxmax()
#     dataitem.data = df[:maxdex].reset_index(drop=True)
#     dataitem.info['max force trim indices'] = (df['index'][0], df['index'][maxdex])
#     return dataitem
#
#
# @processing_function
# def calculate_eng_stress_strain_gradient(dataitem) -> DataItem:
#     data = dataitem.data
#     dataitem.data['eng curve gradient'] = np.gradient(data['eng stress'], data['eng strain'])
#     return dataitem
#
#
# @processing_function
# def calculate_elastic_modulus(dataitem):  # _after_lyp
#     gradient = dataitem.data['eng curve gradient']
#     max_g_idx = gradient.idxmax()
#     dataitem.info['elastic modulus'] = np.average(gradient[max_g_idx + 5:max_g_idx + 15])
#     return dataitem
#
#
# # @processing_function
# # def select_pois_manually(dataitem):
# #     # config
# #     x_key = 'eng strain'
# #     y_key = 'eng stress'
# #     figsize = (12, 8)
# #     data = dataitem.data
# #     poi_list = ["LPL", "UPL", "YS", "UTS"]
# #
# #     x, y = data[x_key], data[y_key]
# #
# #     # plot setup
# #     fig = plt.figure(figsize=figsize)
# #     gs = GridSpec(nrows=2, ncols=2)
# #     c_ax = fig.add_subplot(gs[0, 0])  # clicker ax
# #     c_ax.grid()
# #     v_ax = fig.add_subplot(gs[0, 1])  # viewer ax
# #     v_ax.grid()
# #     tc_ax = fig.add_subplot(gs[1, 0])  # click table ax
# #     tc_ax.axis('off')
# #     td_ax = fig.add_subplot(gs[1, 1])  # data poit table ax
# #     td_ax.axis('off')
# #
# #     # plot data
# #     c_ax.plot(x, y, lw=0, marker='o', alpha=0.2, mfc='none')
# #     c_ax.plot(x, y, color='k')
# #     v_ax.plot(x, y, lw=0, marker='o', alpha=0.2, mfc='none')
# #     v_ax.plot(x, y, color='k')
# #
# #     # plot empty tables
# #     info = pd.Series(index=poi_list)
# #     c_table = pd.DataFrame(info, columns=['Cursor click coords'])
# #     pd.plotting.table(ax=tc_ax, data=c_table, loc='center')
# #     d_table = pd.DataFrame(info, columns=['Data point nearest cursor click'])
# #     pd.plotting.table(ax=td_ax, data=d_table, loc='center')
# #
# #     # setup klicker
# #     zoom_factory(c_ax)
# #     ph = panhandler(fig, button=2)
# #     klicker = clicker(ax=c_ax, classes=poi_list)
# #
# #     def do_on_click(click, poi):
# #         # add click coords to table
# #         prep_tup = lambda tup: str(tuple(map(lambda f: round(f, 4), tup)))
# #         c_table.loc[poi] = prep_tup(click)
# #         tc_ax.clear()
# #         tc_ax.axis('off')
# #         pd.plotting.table(ax=tc_ax, data=c_table, loc='center')
# #
# #         # find nearest datapoint
# #         idx = (np.abs(x - click[0])).argmin()
# #         nearest_point = (x[idx], y[idx])
# #
# #         # add the nearest data point to table
# #         d_table.loc[poi] = prep_tup(nearest_point)
# #         td_ax.clear()
# #         td_ax.axis('off')
# #         pd.plotting.table(ax=td_ax, data=d_table, loc='center')
# #
# #         # plot data point on view ax
# #         v_ax.plot(nearest_point[0], nearest_point[1], label=poi, lw=0, marker='o')
# #         v_ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
# #         plt.tight_layout()
# #
# #     klicker.on_point_added(do_on_click)
# #
# #     # run gui
# #     plt.tight_layout()
# #     plt.show()
# #
# #     # save pois
# #     dataitem.info = dataitem.info.append(d_table['Data point nearest cursor click'])
# #
# #     return dataitem
#
#
# @processing_function
# def calculate_offset_yield_point(dataitem) -> DataItem:
#     return dataitem
#
#
# @processing_function
# def trim_using_considere_criterion(dataitem: DataItem) -> DataItem:
#     df = dataitem.data
#     slope = np.gradient(df['Jaw(mm)'], df['Force(kN)'])
#     maxdex = np.argmin(slope)
#     try:
#         dataitem.data = df[:maxdex].reset_index(drop=True)
#         dataitem.info['considere trim indices'] = (df['index'][0], df['index'][maxdex])
#     except KeyError:
#         dataitem.data = df[:maxdex].reset_index(drop=False)
#         dataitem.info['considere trim indices'] = (df['index'][0], df['index'][maxdex])
#     return dataitem
#
#
# def calculate_engineering_stress_strain(dataitem: DataItem):
#     """ Calculates engineering stress and strain from force and deformation. """
#     info: pd.Series = dataitem.info
#     data: pd.Dataframe = dataitem.data
#     A_0, L_0 = info['A_0'], info['L_0']
#     force, disp = data["Force(kN)"], data["Jaw(kN)"]
#     eng_stress = force/A_0
#     eng_strain = disp/L_0
#     data['eng_stress'] = eng_stress
#     data['eng_strain'] = eng_strain
#     return dataitem
#
#
# def calculate_true_stress_strain(dataitem: DataItem):
#     """ Calculates true stress and strain from force and deformation. """
#     info: pd.Series = dataitem.info
#     data: pd.Dataframe = dataitem.data  # todo: check if this provides a pointer or a copy
#
#     if info['test_type'] == 'tensile test':
#         A_0, L_0 = info['A_0'], info['L_0']
#         force, disp = data["Force(kN)"], data["Jaw(kN)"]
#         eng_stress = force/A_0
#         eng_strain = disp/L_0
#         true_stress = eng_stress*(1 + eng_strain)
#         true_strain = np.log(1 + eng_strain)
#
#     elif info['test_type'] == 'uniaxial compression test':
#         A_0, L_0 = info['A_0'], info['L_0']
#         force, disp = data["Force(kN)"], data["Jaw(kN)"]
#         eng_stress = force/A_0
#         eng_strain = disp/L_0
#         true_stress = eng_stress*(1 - eng_strain)
#         true_strain = -np.log(1 - eng_strain)
#
#     elif info['test_type'] == 'plane strain compression test':
#         A_0, L_0 = info['A_0'], info['L_0']
#         force, disp = data["Force(kN)"], data["Jaw(kN)"]
#         eng_stress = force/A_0
#         eng_strain = disp/L_0
#         true_stress = eng_stress*(1 - eng_strain)
#         true_strain = -np.log(1 - eng_strain)
#
#     else:
#         print(f'Invalid test type "{info["test_type"]}" for dataitem "{dataitem.test_id}"')
#         true_stress = true_strain = None
#
#     data['true_stress'] = true_stress
#     data['true_strain'] = true_strain
#     return dataitem
#
# @processing_function
# def trim_initial_cluster(dataitem, eps=3, min_samples=8):
#     model = DBSCAN(eps=eps, min_samples=min_samples)
#     df = copy.deepcopy(dataitem.data)
#     strain = df['Strain'].values.reshape(-1, 1)
#     stress = df['Stress(MPa)'].values.reshape(-1, 1)
#     X = np.hstack([strain, stress])
#     yhat = model.fit_predict(X)
#     clusters = np.unique(yhat)
#     try:
#         initial_cluster = clusters[1]
#     except IndexError:
#         return dataitem
#     row_ix = np.where(yhat == initial_cluster)
#     clusters = np.unique(yhat)
#     row_ixs = list([np.where(yhat == cluster) for cluster in clusters])
#     min_cluster_idx = pd.Series([np.average(rix) for rix in row_ixs]).idxmin()
#     remove_row_ixs = row_ixs[min_cluster_idx]
#     mindex = max(remove_row_ixs)[-1]
#     dataitem.data = df[mindex:].reset_index(drop=True)
#     try:
#         dataitem.info['cluster trim indices'] = (df['index'][mindex], df['index'].iloc[-1])
#     except KeyError:
#         df = dataitem.data
#         dataitem.info['cluster trim indices'] = (df['Time(sec)'].idxmin(), df['Time(sec)'].idxmax())
#     return dataitem
