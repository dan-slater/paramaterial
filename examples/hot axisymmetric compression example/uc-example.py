import numpy as np
import pandas as pd

from paramaterial.plotting.grid_plots import results_grid_plot
from paramaterial.plug import DataSet, DataItem


path_dict = {
        'screened': {'info_path': 'info/02 screened info.xlsx',
                     'data_dir': 'data/02 screened data'},
        'trimmed': {'info_path': 'info/03 trimmed info.xlsx',
                    'data_dir': 'data/03 trimmed data'},
        'corrected': {'info_path': 'info/04 corrected info.xlsx',
                    'data_dir': 'data/04 corrected data'},
        'representative data': {'info_path': 'info/11 representative info.xlsx',
                                'data_dir': 'data/11 representative data'},
        'representative model': {'info_path': 'info/11 representative info.xlsx',
                                 'data_dir': 'data/13 representative data ramberg'},
        'fitted': {'info_path': 'info/02 screened info.xlsx',
                   'data_dir': 'data/12 fitted data ramberg'},
    }


def make_friction_corrected_data():
    dataset = DataSet()
    dataset.load(**path_dict['trimmed'])

    def correct_for_friction(dataitem: DataItem):
        info = dataitem.info
        data = dataitem.data

        h_0 = info['L_0 (mm)']  # initial height in axial direction
        h = h_0 - data['Jaw(mm)']  # instantaneous height

        d_0 = info['D_0 (mm)']  # initial diameter
        d = d_0*np.sqrt(h_0/h)  # instantaneous diameter

        F = data['Force(kN)']  # force
        P = F*1000*4/(np.pi*d**2)  # pressure (MPa)

        mu = 0.2
        R = P/(1 + (mu*d)/(3*h))  # friction-corrected stress

        dataitem.data['Stress(MPa)'] = R
        dataitem.data['uncorrected stress'] = data['Stress(MPa)']

        return dataitem

    dataset.datamap = map(lambda dataitem: correct_for_friction(dataitem), dataset.datamap)

    dataset.dump(**path_dict['corrected'])


def make_outlier_graphic():
    pass



def make_grid_plots():
    data_keys_dict = {
        'screened': {},
        'trimmed': {},
        'representative': {'x_data_key': 'strain', 'y_data_key': 'mean',
                           'y_upper_key': 'max', 'y_lower_key': 'min'},
        'fitted': {'x_data_key': 'model strain', 'y_data_key': 'ramberg stress'}
    }
    height = 4
    width = 3
    dataset_config = {'reference': ['Baron']}
    row_vals, col_vals = ['H560'], [10]
    col_titles = ['Raw Data']
    results_grid_plot(stage='screened',
                      name=f'baron-raw',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 175},
                      plot_kwargs={'dataset_paths': path_dict['screened'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['screened']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})
    col_titles = ['Trimmed Data']
    results_grid_plot(stage='trimmed',
                      name=f'baron-trimmed',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 175},
                      plot_kwargs={'dataset_paths': path_dict['trimmed'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['trimmed']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})
    col_titles = ['Corrected Data']
    results_grid_plot(stage='trimmed',
                      name=f'baron-corrected',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 175},
                      plot_kwargs={'dataset_paths': path_dict['corrected'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['trimmed']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})
    col_titles = ['Fitted Curves']
    results_grid_plot(stage='fitted',
                      name=f'baron-fitted',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 175},
                      plot_kwargs={'dataset_paths': path_dict['fitted'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['fitted']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})
    col_titles = ['Representative Data']
    results_grid_plot(stage='representative',
                      name=f'baron-repr-data',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 175},
                      plot_kwargs={'dataset_paths': path_dict['representative data'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['representative']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})
    col_titles = ['Representative Models']
    results_grid_plot(stage='representative',
                      name=f'baron-repr-models',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 175},
                      plot_kwargs={'dataset_paths': path_dict['representative model'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['representative']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})


if __name__ == '__main__':
    # make_friction_corrected_data()
    make_grid_plots()