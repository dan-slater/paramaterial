from typing import Dict, List
from paramaterial.screening import make_screening_pdf, copy_screened_data
from paramaterial.plug import DataSet
from paramaterial import proc_ops


def load_dataset(data_dir: str, info_path: str):
    dataset = DataSet()
    dataset.load_data(data_dir=data_dir, info_path=info_path)
    return dataset


def _map_ops(dataset: DataSet, ops: List[callable], ops_dict: Dict[str, callable]) -> DataSet:
    selected_ops = [ops_dict[op_name] for op_name in ops]
    for op in selected_ops:
        # dataset.datamap = map(lambda dataitem: op(dataitem), dataset.datamap)
        dataset.datamap = (lambda dataitem: op(dataitem), dataset.datamap)
    return dataset


def trim_data(dataset: DataSet, ops: List[str]) -> DataSet:
    dataset.datamap = map(lambda dataitem: proc_ops.store_initial_indices(dataitem), dataset.datamap)
    dataset.datamap = map(lambda dataitem: proc_ops.trim_using_sampling_rate(dataitem), dataset.datamap)
    dataset.datamap = map(lambda dataitem: proc_ops.trim_using_max_force(dataitem), dataset.datamap)
    dataset.datamap = map(lambda dataitem: proc_ops.trim_initial_cluster(dataitem), dataset.datamap)
    ops_dict = {
        'sampling rate': proc_ops.trim_using_sampling_rate,
        'max force': proc_ops.trim_using_max_force,
        'initial cluster': proc_ops.trim_initial_cluster
    }
    # dataset = _map_ops(dataset, ops, ops_dict)
    return dataset


def calculate_true_stress_strain():
    pass


def correct_data(dataset: DataSet, ops: List[str]):
    ops_dict = {
        'friction': proc_ops.correct_for_friction
    }
    dataset = _map_ops(dataset, ops, ops_dict)
    return dataset


def make_representative_curves():
    # todo: make curve for whole dataset
    # todo: make curves for permutations
    pass


def read_material_parameters():
    pass


def fit_models():
    pass


def make_test_reports():
    pass


def make_dataset_report():
    pass


