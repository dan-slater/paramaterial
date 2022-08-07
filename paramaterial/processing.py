"""Functions for post-processing material test data. (Stress-strain)"""
from functools import wraps
from typing import Callable, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_interactions import zoom_factory, panhandler
from mpl_point_clicker import clicker

from paramaterial.plug import DataItem


def process_data(dataitem: DataItem, cfg: Dict):
    """ Apply processing functions to a datafile object. """
    processing_operations = [
        calculate_force_disp_from_eng_curve,
        trim_using_max_force,
        calculate_eng_stress_strain_gradient,
        calculate_elastic_modulus,
        calculate_offset_yield_point,
        select_pois_manually,
    ]
    dataitem = store_initial_indices(dataitem)
    for proc_op in processing_operations: # todo: change order of error check
        if proc_op.__name__ in cfg['operations']:
            print(f'{".": <10}Running {proc_op.__name__}.')
            dataitem = proc_op(dataitem)
        else:
            print('!! Processing operation selection error. !!')
    return dataitem


def processing_function(function: Callable[[DataItem], DataItem]):
    """ Applies function to dataitem then returns it. Just returns dataitem if any exception raised. """

    @wraps(function)
    def wrapper(dataitem: DataItem):
        try:
            return function(dataitem)
        except TypeError as e:
            print(e)
            # log_error(e)
            return dataitem

    return wrapper


@processing_function
def store_initial_indices(dataitem):
    df = dataitem.data
    dataitem.info_row['raw data indices'] = (0, len(df))
    dataitem.data = df[:].reset_index(drop=False)
    return dataitem


@processing_function
def calculate_force_disp_from_eng_curve(dataitem: DataItem) -> DataItem:
    e = dataitem.data['eng strain']
    s = dataitem.data['eng stress']
    L_0 = dataitem.info_row['L_0 (mm)']
    A_0 = dataitem.info_row['A_0 (mm^2)']
    dataitem.data['Jaw(mm)'] = e.values * L_0
    dataitem.data['Force(kN)'] = s.values * A_0 * 0.001
    return dataitem


@processing_function
def trim_using_max_force(dataitem):
    df = dataitem.data
    maxdex = df['Force(kN)'].idxmax()
    dataitem.data = df[:maxdex].reset_index(drop=True)
    dataitem.info_row['max force trim indices'] = (df['index'][0], df['index'][maxdex])
    return dataitem


@processing_function
def calculate_eng_stress_strain_gradient(dataitem) -> DataItem:
    data = dataitem.data
    dataitem.data['eng curve gradient'] = np.gradient(data['eng stress'], data['eng strain'])
    return dataitem


@processing_function
def calculate_elastic_modulus(dataitem):  # _after_lyp
    gradient = dataitem.data['eng curve gradient']
    max_g_idx = gradient.idxmax()
    dataitem.info_row['elastic modulus'] = np.average(gradient[max_g_idx + 5:max_g_idx + 15])
    return dataitem


@processing_function
def select_pois_manually(dataitem):
    # config
    x_key = 'eng strain'
    y_key = 'eng stress'
    figsize = (12, 8)
    data = dataitem.data
    poi_list = ["LPL", "UPL", "YS", "UTS"]

    x, y = data[x_key], data[y_key]

    # plot setup
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=2, ncols=2)
    c_ax = fig.add_subplot(gs[0, 0])  # clicker ax
    c_ax.grid()
    v_ax = fig.add_subplot(gs[0, 1])  # viewer ax
    v_ax.grid()
    tc_ax = fig.add_subplot(gs[1, 0])  # click table ax
    tc_ax.axis('off')
    td_ax = fig.add_subplot(gs[1, 1])  # data poit table ax
    td_ax.axis('off')

    # plot data
    c_ax.plot(x, y, lw=0, marker='o', alpha=0.2, mfc='none')
    c_ax.plot(x, y, color='k')
    v_ax.plot(x, y, lw=0, marker='o', alpha=0.2, mfc='none')
    v_ax.plot(x, y, color='k')

    # plot empty tables
    info = pd.Series(index=poi_list)
    c_table = pd.DataFrame(info, columns=['Cursor click coords'])
    pd.plotting.table(ax=tc_ax, data=c_table, loc='center')
    d_table = pd.DataFrame(info, columns=['Data point nearest cursor click'])
    pd.plotting.table(ax=td_ax, data=d_table, loc='center')

    # setup klicker
    zoom_factory(c_ax)
    ph = panhandler(fig, button=2)
    klicker = clicker(ax=c_ax, classes=poi_list)

    def do_on_click(click, poi):
        # add click coords to table
        prep_tup = lambda tup: str(tuple(map(lambda f: round(f, 4), tup)))
        c_table.loc[poi] = prep_tup(click)
        tc_ax.clear()
        tc_ax.axis('off')
        pd.plotting.table(ax=tc_ax, data=c_table, loc='center')

        # find nearest datapoint
        idx = (np.abs(x - click[0])).argmin()
        nearest_point = (x[idx], y[idx])

        # add the nearest data point to table
        d_table.loc[poi] = prep_tup(nearest_point)
        td_ax.clear()
        td_ax.axis('off')
        pd.plotting.table(ax=td_ax, data=d_table, loc='center')

        # plot data point on view ax
        v_ax.plot(nearest_point[0], nearest_point[1], label=poi, lw=0, marker='o')
        v_ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.tight_layout()

    klicker.on_point_added(do_on_click)

    # run gui
    plt.tight_layout()
    plt.show()

    # save pois
    dataitem.info_row = dataitem.info_row.append(d_table['Data point nearest cursor click'])

    return dataitem


@processing_function
def calculate_offset_yield_point(dataitem) -> DataItem:
    return dataitem


@processing_function
def trim_using_considere_criterion(dataitem: DataItem) -> DataItem:
    df = dataitem.data
    slope = np.gradient(df['Jaw(mm)'], df['Force(kN)'])
    maxdex = np.argmin(slope)
    try:
        dataitem.data = df[:maxdex].reset_index(drop=True)
        dataitem.info_row['considere trim indices'] = (df['index'][0], df['index'][maxdex])
    except KeyError:
        dataitem.data = df[:maxdex].reset_index(drop=False)
        dataitem.info_row['considere trim indices'] = (df['index'][0], df['index'][maxdex])
    return dataitem
