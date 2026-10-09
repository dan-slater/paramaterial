# Preparing Data and Info

Paramaterial reads a set of tests from two things:

- a **data folder** with one CSV file per test, named `<test_id>.csv`, all with the same column headers;
- an **info table** (`.xlsx`, `.csv` or `.ods`) with one row per test: a `test_id` column, plus any metadata you want
  to group, filter or colour by, such as material, temperature or strain rate.

After preparation, the project folder looks like this:

```shell
project
├── data
│   ├── 00 raw data
│   │   ├── old_name_1.csv
│   │   └── ...
│   └── 01 prepared data
│       ├── test_01.csv
│       └── ...
├── info
│   └── 01 prepared info.xlsx
└── analysis.ipynb
```

To get there:

1. Create the info table.
2. Format the raw data files and copy them to the prepared data folder.
3. Rename the prepared data files to their test IDs.
4. Run `check_formatting`, then load the `DataSet`.

## Creating the info table

The info table holds everything you know about each test. The first column is the test ID; the rest are the metadata
you need. For example:

| test_id | material | temperature | rate  |
|---------|----------|-------------|-------|
| test_01 | Steel    | 20          | 0.001 |
| test_02 | Steel    | 100         | 0.001 |
| test_03 | Steel    | 20          | 0.01  |

Build it by hand in a spreadsheet, or with pandas from your file names. Save it in the `info` folder. When you rename
files in step 3, it is convenient to keep an `old filename` column in this table, or in a separate rename table, that
maps each raw file name (without `.csv`) to its test ID.

## Formatting the data files

Raw files from a test machine often have headers split over several rows, inconsistent names, or a delimiter other
than a comma. For example, this tab-separated file:

| Time | Total Strain (mm/mm) | True Stress |
|------|----------------------|-------------|
| (s)  |                      | (MPa)       |
| 0    | 0.0000               | 0.2         |
| 1    | 0.0013               | 10.7        |

`format_data_file_contents` combines the header rows into one, renames columns, and writes every CSV in the raw folder
to the prepared folder with comma delimiters:

```python
import paramaterial as pam

pam.format_data_file_contents('data/00 raw data', 'data/01 prepared data',
                              header_rows=[0, 1], delimiter='\t',
                              header_rename_dict={'Total Strain (mm/mm)': 'Strain',
                                                  'True Stress (MPa)': 'Stress_MPa'})
```

The result:

| Time (s) | Strain | Stress_MPa |
|----------|--------|------------|
| 0        | 0.0000 | 0.2        |
| 1        | 0.0013 | 10.7       |

`Strain` and `Stress_MPa` are the default column names of the processing functions. Other names work too; pass them
as `strain_key=` and `stress_key=`.

## Renaming the data files

`rename_data_files` renames `<old filename>.csv` to `<test_id>.csv` in place, using a table (a path to `.xlsx` or
`.csv`, or a DataFrame) with both columns:

```python
pam.rename_data_files('data/01 prepared data', 'info/00 rename list.xlsx',
                      old_filename_col='old filename', test_id_col='test_id')
```

To keep the raw folder untouched and copy and rename in one step, `copy_data_and_rename_by_test_id(data_in, data_out,
info_table)` does the same from an info table with `old_filename` (including `.csv`) and `test_id` columns.

## Checking the formatting

`check_formatting` reports any data file whose headers differ from the first file, and raises an error if two files
have identical contents, which is a common copy-paste mistake:

```python
pam.check_formatting('data/01 prepared data', 'info/01 prepared info.xlsx')
```

```text
Headers are the same in all files as in the first file.
Consistent headers are: Time (s), Strain, Stress_MPa
No duplicate files (by hashing contents) found in "data/01 prepared data".
```

Columns that are allowed to differ between files can be listed with `skip_headers=[...]`.

## Loading the DataSet

```python
ds = pam.DataSet('info/01 prepared info.xlsx', 'data/01 prepared data')
```

Each test is now a `DataItem` with `.data` (the CSV as a DataFrame) and `.info` (its row of the info table), ready for
the processing functions. See the [basic usage notebook](https://github.com/dan-slater/paramaterial-examples/blob/main/examples/dan-msc-basic-usage/dan-msc-basic-usage-0.1.0.ipynb)
for a full analysis.
