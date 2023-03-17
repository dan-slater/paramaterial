# How-to guides

This section contains a collection of how-to guides for common tasks.

## How to use the DataSet class

Create a folder containging all the .csv files you want to load. Additionally, create an excel spreadsheet or csv file containing all the metadata for the files, where each row corresponds to a file.

Then, load the dataset using the following code:

```python
from paramaterial import DataSet
dataset = DataSet("path/to/folder", "path/to/metadata.csv")
```

The metadata is stored in the `dataset.info_table` property. You can get a dataframe containing the info table, or update the info table using the following code:

```python
# Get the info table as a pandas dataframe
info_table = dataset.info_table
# Update the info table
dataset.info_table = info_table
```


## How to make representative curves

The paramaterial.modelling module contains a 'make_representative_data' function that can be used to make representative curves from a dataset. The functino can be used as follows:

```python
import paramaterial as pam
from paramaterial import DataSet

dataset = DataSet("path/to/folder", "path/to/metadata.csv")
representative_curves = pam.modelling.make_representative_data(dataset, "path/to/output_folder",
                                                               "path/to/output_metadata.csv", repres_col="Stress_MPa")
```
```