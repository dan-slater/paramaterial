"""Make all plots in report from this file."""

from paramaterial.plotting.grid_plots import results_grid_plot


def main():
    make_data_grid_plots(
        stage='screened',
        baron=True,
        # hyde=True
    )
    # make_data_grid_plots(
    #     stage='trimmed',
    #     # hyde=True,
    #     # baron=True,
    # )
    # make_data_grid_plots(
    #     stage='representative',
    #     # hyde=True,
    #     # baron=True,
    # )
    # make_data_grid_plots(
    #     stage='fitted',
    #     # hyde=True,
    #     # baron=True,
    # )



def make_data_grid_plots(stage: str, baron: bool = False, hyde: bool = False, aakash: bool = False):
    path_dict = {
        'screened': {'info_path': '../../info/02 screened info.xlsx',
                     'data_dir': '../../data/02 screened data'},
        'trimmed': {'info_path': '../../info/02 screened info.xlsx',
                      'data_dir': '../../data/03 trimmed data'},
        'representative data': {'info_path': '../../info/11 representative info.xlsx',
                                 'data_dir': '../../data/11 representative data'},
        'representative model': {'info_path': '../../info/11 representative info.xlsx',
                           'data_dir': '../../data/11 representative data ramberg'},
        'fitted': {'info_path': '../../info/02 screened info.xlsx',
                   'data_dir': '../../data/12 fitted data ramberg'},
    }
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
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 225},
                      plot_kwargs={'dataset_paths': path_dict['screened'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['screened']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})
    col_titles = ['Processed Data']
    results_grid_plot(stage='trimmed',
                      name=f'baron-processed',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 225},
                      plot_kwargs={'dataset_paths': path_dict['trimmed'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['trimmed']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})
    col_titles = ['Fitted Curves']
    results_grid_plot(stage='fitted',
                      name=f'baron-fitted',
                      grid_kwargs={'height': height, 'width': width,
                                   'rows': len(row_vals), 'cols': len(col_vals), 'col_titles': col_titles},
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 225},
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
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 225},
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
                      axis_kwargs={'x_sep': 0.3, 'y_sep': 50, 'x_max': 1.35, 'y_max': 225},
                      plot_kwargs={'dataset_paths': path_dict['representative model'], 'dataset_config': dataset_config,
                                   'row_vals': row_vals, 'col_vals': col_vals,
                                   'rows_key': 'material', 'cols_key': 'rate',
                                   **data_keys_dict['representative']},
                      legend_kwargs={'min_T': 300, 'max_T': 500})


def make_params_grid_plots(model: str, baron: bool = False, hyde: bool = False, aakash: bool = False):
    path_dict = {
        'ramberg': {'info_path': '../info/12 fitted params info.xlsx',
                     'data_dir': '../data/12 fitted params data'},
    }
    row_keys_dict = {'ramberg': ['C', 'q'], 'voce': ['s_u', 'd']}
    y_labels_dict = {'ramberg': ['C (MPa)', 'n'], 'voce': ['s_u', 'd']}
    if hyde:
        dataset_config = {'reference': ['Hyde']}
        col_vals = [10, 30, 100]
        col_titles = [r'$\dot{\varepsilon}$ = ' + str(s) + ' s$^{-1}$' for s in col_vals]
        line_vals = ['PSC', 'PSC*']
        line_styles = {'PSC':{'ls': '-', 'marker':'s'},
                       'PSC*': {'ls': '-', 'marker':'*'}}
        results_grid_plot(stage=model,
                          name=f'hyde-{model}-params-grid',
                          grid_kwargs={'height': height, 'width': width,
                                       'rows': 2, 'cols': 3,
                                       'col_titles': col_titles},
                          axis_kwargs={'x_sep': 100, 'y_seps': [20,0.1], 'x_max': 550, 'y_maxs': [110,1.05],
                                       'y_labels': y_labels_dict[model]},
                          plot_kwargs={'dataset_paths': path_dict[model], 'dataset_config': dataset_config,
                                       'row_data_keys': row_keys_dict[model], 'cols_key': 'rate', 'col_vals': col_vals,
                                       'lines_key': 'test type', 'line_vals': line_vals, 'line_styles': line_styles},
                          legend_kwargs={})
    if baron:
        dataset_config = {'reference': ['Baron']}
        col_vals = [1, 10, 50, 100]
        col_titles = [r'$\dot{\varepsilon}$ = ' + str(s) + ' s$^{-1}$' for s in col_vals]
        lines_key = 'material'
        line_vals = ['AC', 'H560', 'H580']
        line_styles = {'AC': {'ls': '-', 'marker': 'x'},
                       'H560': {'ls': '-', 'marker': 'o'},
                       'H580': {'ls': '-', 'marker': 'd'}}
        results_grid_plot(stage=model,
                          name=f'baron-{model}-params-grid',
                          grid_kwargs={'height': height, 'width': width,
                                       'rows': 2, 'cols': 4,
                                       'col_titles': col_titles},
                          axis_kwargs={'x_sep': 100, 'y_seps': [20, 0.1], 'x_max': 550, 'y_maxs': [110, 1.05],
                                       'y_labels': y_labels_dict[model]},
                          plot_kwargs={'dataset_paths': path_dict[model], 'dataset_config': dataset_config,
                                       'row_data_keys': row_keys_dict[model], 'cols_key': 'rate', 'col_vals': col_vals,
                                       'lines_key': lines_key, 'line_vals': line_vals, 'line_styles': line_styles},
                          legend_kwargs={})




if __name__ == '__main__':
    main()
