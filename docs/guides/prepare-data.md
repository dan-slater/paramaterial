# Preparing Data and Info

After preparation, we want to have a directory structure that looks like this:

```shell
project
├── data
    ├── raw
        ├── old_name_1.csv
        ├── old_name_2.csv
        └── ...
    ├── prepared
        ├── test_id_1.csv
        ├── test_id_2.csv
        └── ...
├── info
    ├── prepared info.xlsx
    └── ...
├── repeatable-analysis.ipynb
└── ...
```
    
To get there, we need to:

1. Create the info table
2. Prepare the data files
3. Run the `check_formatting` function to check that the data files are formatted correctly and match the test IDs in the info table.

## Creating the info table

   1. Create a single spreadsheet for the data
   2. The first column should be the test ID
   3. Other columns should be the info we need for each test
   4. Save the spreadsheet as an Excel file in the `info` directory

The info table is a spreadsheet that contains all the information we need for each test. It should have a column for the test ID, and then other columns for the info we need for each test. For example, if we need the test ID, the material name, the test temperature, and the test speed, the info table would look like this:

| Test ID | Material | Temperature (C) | Speed (mm/min) | ... |
|---------|----------|-----------------|----------------| --- |
| test_1  | Steel    | 20              | 10             | ... |
| test_2  | Steel    | 100             | 10             | ... |
| test_3  | Steel    | 20              | 20             | ... |
| test_4  | Steel    | 100             | 20             | ... |
| ...     | ...      | ...             | ...            | ... |
   
This step can be done manually or using scripting, depending on the shape of your data.

Once you have gathered all your metadata (information about the tests) into a spreadsheet, save it as an Excel file in the `info` directory.
A CSV file can also be used.

## Preparing the data files

1. Gather the CSV data files into a directory called `00 raw data`.
2. Format the data file contents and copy to `01 prepared data`.
3. Rename the data files in `01 prepared data` according to the test IDs.

The `format_data_file_headers` function can be used format the headers, change the delimiter, and copy the raw data into the prepared data folder.

Consider a raw data file with multi-level headers and inconsistent naming:

| Time  | Total Strain (mm/mm)  | True Stress | ... |
|----------|-----------------------|-------------| --- |
| (s)      |                       | (MPa)       |    |
| 0        | 0.0000                | 0.2         | ... |
| 1        | 0.0013                | 10.7        | ... |
| 2        | 0.0022                | 21.3        | ... |
| ...     | ...      | ...             | ...            | 

We can run the `format_data_file_headers` function to standardise the headers:

```python
from paramaterial import format_data_file_contents
format_data_file_contents('data/00 raw data', 'data/01 prepared data',
                           header_rename_dict={'Total Strain (mm/mm)': 'Strain',
                                               'True Stress (MPa)': 'Stress (MPa)'},
                           header_rows=[0, 1], delimiter='\t')
```
This combines the header rows into a single row, and renames specified headers. 
This also allows us to change the delimiter of the file. The resulting data file will look like this:

| Time (s) | Strain | Stress (MPa) | ... |
|----------|--------|--------------| --- |
| 0        | 0.0000 | 0.2          | ... |
| 1        | 0.0013 | 10.7         | ... |
| 2        | 0.0022 | 21.3         | ... |
| ...     | ...      | ...             | ...            | 

This will be done for every data file in the raw data folder.

### Renaming the data files

The data files can now be renamed according to the test ID using `rename_data_files`:

```python
from paramaterial import rename_data_files
rename_data_files(data_dir='data/01 prepared data', rename_table='info/00 rename list.xlsx',
                  old_filename_col='old filename', test_id_col='test id')
```

This will rename the CSV files in `data_dir` from `<old filename>.csv` to `<test id>.csv`.

### Running the formatting check

The `check_formatting` function can be used:

```python

```

I
