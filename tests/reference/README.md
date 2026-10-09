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

To check the current package against the reference (needs `nbclient`, `ipykernel` and a LaTeX install with amsmath):

    git clone https://github.com/dan-slater/paramaterial-examples .examples
    python tests/reference/run_examples.py --examples .examples/examples --out /tmp/pam-new \
        --compare tests/reference/examples --rtol 1e-6
