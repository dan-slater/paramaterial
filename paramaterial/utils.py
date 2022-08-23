# todo: import utility functions from this module

""" Engineering calculations for load and stress. [danslater, 1march2022] """
import numpy as np

from plug import DataItem
import pandas as pd

TEST_TYPES = ('tensile test', 'uniaxial compression test', 'plane strain compression test')


def calculate_engineering_stress_strain(dataitem: DataItem):
    """ Calculates engineering stress and strain from force and deformation. """
    info: pd.Series = dataitem.info
    data: pd.Dataframe = dataitem.data
    if info['test_type'] in TEST_TYPES:
        A_0, L_0 = info['A_0'], info['L_0']
        force, disp = data["Force(kN)"], data["Jaw(kN)"]
        eng_stress = force / A_0
        eng_strain = disp / L_0
        data['eng_stress'] = eng_stress
        data['eng_strain'] = eng_strain
    else:
        print(f'Invalid test type "{info["test_type"]}" for dataitem "{dataitem.test_id}"')
        print(f'Test type must be one of: {TEST_TYPES}')  # todo: move this error check to decorator
    return dataitem


def calculate_true_stress_strain(dataitem: DataItem):
    """ Calculates true stress and strain from force and deformation. """
    info: pd.Series = dataitem.info
    data: pd.Dataframe = dataitem.data  # todo: check if this provides a pointer or a copy

    if info['test_type'] == 'tensile test':
        A_0, L_0 = info['A_0'], info['L_0']
        force, disp = data["Force(kN)"], data["Jaw(kN)"]
        eng_stress = force / A_0
        eng_strain = disp / L_0
        true_stress = eng_stress * (1 + eng_strain)
        true_strain = np.log(1 + eng_strain)

    elif info['test_type'] == 'uniaxial compression test':
        A_0, L_0 = info['A_0'], info['L_0']
        force, disp = data["Force(kN)"], data["Jaw(kN)"]
        eng_stress = force / A_0
        eng_strain = disp / L_0
        true_stress = eng_stress * (1 - eng_strain)
        true_strain = -np.log(1 - eng_strain)

    elif info['test_type'] == 'plane strain compression test':
        A_0, L_0 = info['A_0'], info['L_0']
        force, disp = data["Force(kN)"], data["Jaw(kN)"]
        eng_stress = force / A_0
        eng_strain = disp / L_0
        true_stress = eng_stress * (1 - eng_strain)
        true_strain = -np.log(1 - eng_strain)

    else:
        print(f'Invalid test type "{info["test_type"]}" for dataitem "{dataitem.test_id}"')
        true_stress = true_strain = None

    data['true_stress'] = true_stress
    data['true_strain'] = true_strain
    return dataitem

# todo: remove duplicated code
# todo: implement PSC formulas
