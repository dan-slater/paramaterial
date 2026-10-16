# Changelog

## 0.2.0 (2026-10-16)

### Plotting

- `dataset_plot` has new legend options: `leg_loc` (default `'best'`), `leg_handletextpad`, `leg_labelspacing`, and
  `leg_outside=True`, which puts the legend on the figure, right of the axes.
- **Breaking:** `dataset_plot`'s `handletextpad` and `labelspacing` arguments are renamed to `leg_handletextpad` and
  `leg_labelspacing`.
- **Changed:** importing paramaterial no longer restyles matplotlib globally. 0.1.x applied `seaborn-dark`, a serif
  font at size 9 and grid lines on import. To get the old look, set these in your own notebook or matplotlibrc.
- `Styler` uses `color_by_label` exactly as given. It is no longer title-cased.
- `dataset_subplots` styles its default `Styler` to the dataset when you don't pass one.
- The `dataset_plot` legend and the shared `dataset_subplots` legend are drawn with matplotlib's default frame.

### Preparing data

- New `format_data_file_contents` combines multi-row headers (leaving out blank cells), renames columns and changes
  the delimiter for a folder of raw CSV files.
- New `rename_data_files` renames data files to their test IDs from a rename table (xlsx, csv or DataFrame).
- `check_formatting(data_dir, info_path=None, skip_headers=None)` reports header mismatches instead of raising, and
  checks for duplicate files. Passing a `DataSet`, as in 0.1.x, still runs the 0.1.x checks.
- `check_column_headers`, `check_for_duplicate_files`, `copy_data_and_rename_by_test_id` and
  `convert_gleeble_output_files_to_csv` are kept from 0.1.x. The MSc example notebooks use them.
- New how-to guide: [Preparing data and info](https://dan-slater.github.io/paramaterial/guides/prepare-data/).

### Compatibility with current pandas, numpy, scipy and matplotlib

paramaterial now runs on Python 3.9 to 3.12 with pandas 1.5 to 3.0, numpy 1.24 to 2.5, scipy 1.10 to 1.18 and
matplotlib 3.7 to 3.11.

- Loading or editing a `DataSet` no longer fails on pandas 3. `pd.to_numeric(errors='ignore')` was removed there.
- **Fixed:** `DataSet.subset` returned the whole DataSet, unfiltered, under numpy 2 when filtering on numeric info
  values. It only printed an error. This silently made every representative curve from `make_representative_data` an
  average of all tests. Filtering errors now raise.
- `make_quality_matrix` works on pandas 2.2 and later.
- `Styler` works on matplotlib 3.9 and later (`plt.cm.get_cmap` was removed).
- PDF screening and test receipts use `pypdf`; `PyPDF2` is no longer maintained.
- `TestReceipts(jinja_env_kwargs=...)` no longer fails.
- `download_example` extracts archives with the tarfile `'data'` filter where Python supports it.

### Reproducibility

The MSc thesis example notebooks (basic usage and case studies CS1 to CS4 in
[paramaterial-examples](https://github.com/dan-slater/paramaterial-examples)) were run on 0.1.3 and on 0.2.0, and
every table they produce was compared (`tests/reference/`). On the thesis-era dependencies the results are identical.
On current dependencies every measured property agrees to 1e-11. Fitted Ramberg-Osgood and Zener-Hollomon parameters
move in the 6th significant figure with the scipy version; the thesis reports them to 3 or 4.

### Packaging

- `pyproject.toml` (PEP 621, setuptools) replaces `setup.py` and `requirements.txt`. Requires Python 3.9 or later.
- Dependencies are what the package imports, with tested lower bounds and no upper bounds. `setuptools`, `chardet`
  and `PyPDF2` are dropped; `pypdf` is added.
- Added `CITATION.cff`, which cites the MSc thesis.
- Tests run on GitHub Actions for Python 3.9 to 3.12.
- Docs: `mkdocs build --strict` passes; example notebooks are linked from the examples repository; the hijacked
  `polyfill.io` script is removed from the site.

## 0.1.3 (2023-09-02) and earlier

Released to PyPI from the MSc work: 0.1.0 to 0.1.2 on 2023-08-15 and 0.1.3 on 2023-09-02. 0.1.3 dropped the `var_`
prefix on variable columns in `ModelSet.fitting_table`, added the `fitting_results` property, and made `predict`
default to the overall data range. See the git history for detail.
