"""Module for obtaining material parameters from datasets."""
from matplotlib import pyplot as plt

from paramaterial.plug import DataSet, DataItem
import paramaterial as pam


def main():
    """Main function."""

    # load processed data
    proc_set = DataSet('data/02 processed data', 'info/02 processed info.xlsx')

    # make representative data from processed data
    pam.processing.make_representative_curves(
        proc_set, 'data/03 representative curves', 'info/03 representative info.xlsx',
        repr_col='Stress(MPa)', repr_by_cols=['material', 'temperature', 'rate'],
        interp_by='Strain'
    )

    # load representative data
    repr_set = DataSet('data/03 representative curves', 'info/03 representative info.xlsx', 'repr id')

    # setup screening plot
    color_by = 'temperature'
    color_norm = plt.Normalize(vmin=proc_set.info_table[color_by].min(), vmax=proc_set.info_table[color_by].max())

    def screening_plot(di: DataItem) -> None:
        """Plot function for screening."""
        plt.plot(di.data['Strain'], di.data['Stress(MPa)'], color=plt.cm.viridis(color_norm(di.info[color_by])))
        plt.xlabel('Strain')
        plt.ylabel('Stress (MPa)')
        plt.title(f'{di.info["material"]}, {di.info["temperature"]}$^{{\circ}}$C, {di.info["rate"]} s$^{{-1}}$')

    # make screening pdf
    pam.processing.make_screening_pdf(

    # screen representative data
    pam.processing.screen_data(
        repr_set, 'data/03 representative curves screening marked 21Oct18h57.pdf',
        'data/screened/12 screened repr data', 'info/screened/12 screened repr info.xlsx'
    )

    # load screened representative data
    screened_repr_set = DataSet('data/screened/12 screened repr data', 'info/screened/12 screened repr info.xlsx')

    # make material parameters from screened representative data
    pam.processing.make_material_parameters(
        screened_repr_set, 'data/04 material parameters', 'info/04 material parameters info.xlsx',
        param_by_cols=['material', 'temperature', 'rate'], param_by='repr id',
        param_func=pam.modelling.hyperelasticity
    )

    # load material parameters
    param_set = DataSet('data/04 material parameters', 'info/04 material parameters info.xlsx')

def make_material_parameters(
        data_set: DataSet, data_dir: str, info_file: str, param_by_cols: List[str], param_by: str,
        param_func: Callable[[DataItem], Dict[str, float]]
) -> None:
    """Make material parameters from data.

    Parameters
    ----------
    data_set : DataSet
        The data set to make material parameters from.
    data_dir : str
        The directory to save the material parameter data in.
    info_file : str
        The file to save the material parameter info in.
    param_by_cols : List[str]
        The columns to group the data by.
    param_by : str
        The column to use for the parameter id.
    param_func : Callable[[DataItem], Dict[str, float]]
        The function to use to make the material parameters.

    """
    # get the parameter ids
    param_ids = data_set.info_table.groupby(param_by_cols)[param_by].unique().apply(lambda x: x[0])

    # make the data and info tables
    data_table = pd.DataFrame(index=param_ids, columns=param_func(data_set.data_table.iloc[0]).keys())
    info_table = data_set.info_table.groupby(param_by_cols).first().loc[param_ids]

    # make the material parameters
    for param_id in param_ids:
        data_table.loc[param_id] = param_func(data_set.data_table.loc[param_id])

    # save the material parameters
    data_table.to_csv(os.path.join(data_dir, 'data.csv'))
    info_table.to_excel(info_file)

def hyperelasticity(data_item: DataItem) -> Dict[str, float]:
    """Make hyperelastic material parameters from data.

    Parameters
    ----------
    data_item : DataItem
        The data item to make material parameters from.

    Returns
    -------
    Dict[str, float]
        The material parameters.

    """
    # get the data
    data = data_item.data

    # make the material parameters
    return {
        'E': data['Stress(MPa)'].max() / data['Strain'].max(),
        'nu': 0.5
    }

# take a dataset and a function to predict data in a dataitem
# take a list of columns to group by
# take a column to use for the parameter id
# make a data table with the parameter ids as the index
# make an info table with the parameter ids as the index
# make the material parameters
# save the material parameters
def fit_material_parameters(
        data_set: DataSet, data_dir: str, info_file: str, param_by_cols: List[str], param_by: str,
        param_func: Callable[[DataItem], Dict[str, float]]
) -> None:
    """Fit material parameters to data.

    Parameters
    ----------
    data_set : DataSet
        The data set to fit material parameters to.
    data_dir : str
        The directory to save the material parameter data in.
    info_file : str
        The file to save the material parameter info in.
    param_by_cols : List[str]
        The columns to group the data by.
    param_by : str
        The column to use for the parameter id.
    param_func : Callable[[DataItem], Dict[str, float]]
        The function to use to fit the material parameters.

    """
    # get the parameter ids
    param_ids = data_set.info_table.groupby(param_by_cols)[param_by].unique().apply(lambda x: x[0])

    # make the data and info tables
    data_table = pd.DataFrame(index=param_ids, columns=param_func(data_set.data_table.iloc[0]).keys())
    info_table = data_set.info_table.groupby(param_by_cols).first().loc[param_ids]

    # make the material parameters
    for param_id in param_ids:
        data_table.loc[param_id] = param_func(data_set.data_table.loc[param_id])

    # save the material parameters
    data_table.to_csv(os.path.join(data_dir, 'data.csv'))
    info_table.to_excel(info_file)

def hyperelasticity(data_item: DataItem) -> Dict[str, float]:
    """Make hyperelastic material parameters from data.

    Parameters
    ----------
    data_item : DataItem
        The data item to make material parameters from.

    Returns
    -------
    Dict[str, float]
        The material parameters.

    """
    # get the data
    data = data_item.data

    # make the material parameters
    return {
        'E': data['Stress(MPa)'].max() / data['Strain'].max(),
        'nu': 0.5
    }

# Path: paramaterial\processing.py
"""Module for processing datasets."""

if __name__ == '__main__':
    main()

# Path: paramaterial\plug.py
"""Module for loading and saving data and info."""