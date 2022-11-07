from matplotlib import pyplot as plt

import paramaterial as pam
from paramaterial.plug import DataSet, DataItem


def main():
    """Main function."""

    def ds_plot(dataset: DataSet, **kwargs) -> plt.Axes:
        return pam.plotting.dataset_plot(
            dataset, x='Strain', y='Stress(MPa)', color_by='temperature',
            cbar=True, cbar_label='Temperature ($^{\circ}$C)',
            xlim=(-0.2, 1.5), grid=True, **kwargs
        )

    proc_set = DataSet('data/02 processed data', 'info/02 processed info.xlsx')

    def multiply_neg_one(di: DataItem) -> DataItem:
        di.data['Stress_MPa'] *= -1
        return di

    proc_set = proc_set.apply(multiply_neg_one)

    def subset_test():
        a = proc_set[{'lot': ['A']}]

    def subset_test_2():
        a = proc_set.get_subset_2({'lot': ['A']})

    def subset_test_3():
        a = proc_set.get_subset_2({'lot': ['A']})


    # function to time a function
    def time_function(func, *args, **kwargs):
        import time
        start = time.time()
        for i in range(50):
            func(*args, **kwargs)
        end = time.time()
        return end - start

    # time the functions
    print('subset_test', time_function(subset_test))
    print('subset_test_2', time_function(subset_test_2))
    print('subset_test_3', time_function(subset_test_3))

    # a = proc_set[{'material': ['AC']}]
    # ds_plot(proc_set.get_subset({'material': ['AC']}))
    # plt.show()



if __name__ == '__main__':
    main()