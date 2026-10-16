# Reference outputs

Numbers recorded from **paramaterial 0.1.3** (the PyPI sdist, identical to commit `db76f84` apart from
`__version__`) on the dependency versions the MSc thesis notebooks print: pandas 1.5.3, seaborn 0.12.2,
matplotlib 3.7.0, with numpy 1.24.4 and scipy 1.10.1, Python 3.11, macOS arm64. Recorded 2026-10-09.

- `modelling.json`: the Ramberg-Osgood fit in `tests/test_modelling.py`. Checked by pytest.
- `examples/*.json`: every DataSet, ModelSet and DataFrame left in the kernel after running the
  [paramaterial-examples](https://github.com/dan-slater/paramaterial-examples) notebooks (basic usage, CS1 to CS4),
  plus the spreadsheets and LaTeX tables they write and their printed text output.

Against the outputs stored in the notebooks from the 2023 thesis run, the reference matches exactly for all
processing results (moduli, proportional limits, proof stresses, UTS, fracture points, rejected specimens,
representative curves, Zener-Hollomon fits). The Ramberg-Osgood parameters agree to about 7 significant figures.
The last digits come from the optimiser and move with scipy version and platform.

The notebooks number their tests in `os.listdir()` order. The thesis ran on Windows, which lists names sorted, so
`run_examples.py` sorts `os.listdir` before running a notebook. Without that, CS1 rejects different specimens on
macOS and Linux.

## Comparison policy

`run_examples.py --compare` requires every value to match to a relative 1e-9. Strings, and the cells of the LaTeX
tables, must be equal; whitespace and `\_` escaping, which pandas changed in `to_latex`, are ignored. The one
exception is numbers that come out of a scipy optimiser. Those are the Ramberg-Osgood C/H and n from `ModelSet`, the
Zener-Hollomon `curve_fit` columns (`lnZ_fit*`, `ZH_fit*`), statistics over those, and curves predicted from fitted
parameters. They are held to 1e-3. See `OPTIMISER_COLUMN` and `OPTIMISER_PATH`.

Why: the fit objective agrees to about 1e-10 on every scipy tried (1.10 to 1.18), so the optimiser finds the same
minimum. Where L-BFGS-B stops along the flat C-n valley moves in the 6th significant figure (worst 2e-5). A standard
deviation over a tight spread of fitted parameters amplifies that to 3e-4. The thesis reports these to 3 or 4
significant figures.

Results on 2026-10-09 (paramaterial 0.2.0):

| Stack | Non-optimiser values | Optimiser outputs |
| --- | --- | --- |
| Python 3.11, pandas 1.5.3, numpy 1.24.4, scipy 1.10.1, matplotlib 3.7.0 | bit-identical | bit-identical |
| Python 3.12, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2 | within 3.6e-10 (fit `error`), all measured properties within 2.6e-12 | within 3.2e-4 (`std_n`), parameters within 2.1e-5 |

To check the current package against the reference (needs `nbclient`, `ipykernel` and a LaTeX install with amsmath):

    git clone https://github.com/dan-slater/paramaterial-examples .examples
    python tests/reference/run_examples.py --examples .examples/examples --out /tmp/pam-new \
        --compare tests/reference/examples

The runner prepends a cell to each notebook that sorts `os.listdir` (see above), puts amsmath in the LaTeX preamble
(the figure cells use `\text{}` with `text.usetex`), and restores `matplotlib.cm.get_cmap` (removed in matplotlib 3.9,
called by CS4's own code).
