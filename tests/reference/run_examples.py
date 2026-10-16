"""Execute the paramaterial-examples notebooks and dump their numbers to JSON.

Used to pin the MSc thesis results (CS1-CS4 + basic usage) across dependency upgrades.

    python tests/reference/run_examples.py --examples .examples/examples --out tests/reference/examples
    python tests/reference/run_examples.py --examples .examples/examples --out /tmp/new --compare tests/reference/examples

Each notebook is run in a scratch copy of its folder, after a prelude cell (PRELUDE_CELL). A final cell is appended
that walks the kernel's globals and records every DataSet (info table + per-item data summary), ModelSet (fitting
table) and DataFrame. Spreadsheets and LaTeX tables the notebook writes into info/ are recorded too. See README.md in
this folder for the comparison policy.
"""
import argparse
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS = {
    'basic-usage': 'dan-msc-basic-usage/dan-msc-basic-usage-0.1.0.ipynb',
    'cs1': 'dan-msc-case-studies/CS1/dan-msc-cs1-0.1.0.ipynb',
    'cs2': 'dan-msc-case-studies/CS2/dan-msc-cs2-0.1.0.ipynb',
    'cs3': 'dan-msc-case-studies/CS3/dan-msc-cs3-0.1.0.ipynb',
    'cs4': 'dan-msc-case-studies/CS4/dan-msc-cs4-0.1.0.ipynb',
}

# Some figure cells switch on text.usetex and use \text{}, which needs amsmath in the LaTeX preamble.
# The notebooks number tests in os.listdir() order. The thesis ran on Windows, where NTFS lists names sorted; sorting
# here makes the test IDs (and so the specimens rejected by the marked screening PDF) the same on every platform.
PRELUDE_CELL = r'''
import matplotlib as _mpl
_mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import os as _os
_listdir = _os.listdir
_os.listdir = lambda path='.': sorted(_listdir(path))
# CS4's own code calls plt.cm.get_cmap, which matplotlib 3.9 removed.
import matplotlib.cm as _cm
if not hasattr(_cm, 'get_cmap'):
    _cm.get_cmap = _mpl.colormaps.get_cmap
'''

DUMP_CELL = r'''
import json as _json, glob as _glob, os as _os
import numpy as _np, pandas as _pd
import paramaterial as _pam

def _val(v):
    if isinstance(v, (bool, _np.bool_)):
        return bool(v)
    if isinstance(v, (int, _np.integer)):
        return int(v)
    if isinstance(v, (float, _np.floating)):
        return None if _np.isnan(v) else float(v)
    if v is None or (not isinstance(v, str) and _pd.isna(v)):
        return None
    return str(v)

def _frame(df):
    df = df.reset_index(drop=True)
    return {'columns': [str(c) for c in df.columns],
            'rows': [[_val(v) for v in row] for row in df.itertuples(index=False)]}

def _data_summary(ds):
    out = {}
    for di in ds:
        num = di.data.select_dtypes('number')
        out[str(di.test_id)] = {'n': int(len(di.data)), 'columns': [str(c) for c in di.data.columns],
                                'sum': {str(c): _val(num[c].sum()) for c in num.columns},
                                'first': {str(c): _val(num[c].iloc[0]) for c in num.columns} if len(num) else {},
                                'last': {str(c): _val(num[c].iloc[-1]) for c in num.columns} if len(num) else {}}
    return out

_dump = {'datasets': {}, 'modelsets': {}, 'frames': {}, 'files': {}}
for _name, _obj in list(globals().items()):
    if _name.startswith('_'):
        continue
    if isinstance(_obj, _pam.DataSet):
        _dump['datasets'][_name] = {'info_table': _frame(_obj.info_table), 'data': _data_summary(_obj)}
    elif isinstance(_obj, _pam.ModelSet):
        _dump['modelsets'][_name] = {'fitting_table': _frame(_obj.fitting_table)}
    elif isinstance(_obj, _pd.DataFrame):
        _dump['frames'][_name] = _frame(_obj)
for _f in sorted(_glob.glob('info/*.xlsx')):
    _dump['files'][_f] = _frame(_pd.read_excel(_f))
for _f in sorted(_glob.glob('info/*.tex')):
    _dump['files'][_f] = open(_f).read()
with open('__dump__.json', 'w') as _fh:
    _json.dump(_dump, _fh, indent=1, sort_keys=True)
'''


def run_notebook(examples: Path, rel: str) -> dict:
    src = examples / rel
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / src.parent.name
        shutil.copytree(src.parent, work)
        nb = nbformat.read(work / src.name, as_version=4)
        nb.cells.insert(0, nbformat.v4.new_code_cell(PRELUDE_CELL))
        nb.cells.append(nbformat.v4.new_code_cell(DUMP_CELL))
        try:
            NotebookClient(nb, timeout=1800, kernel_name='python3',
                           resources={'metadata': {'path': str(work)}}).execute()
        except CellExecutionError as e:
            return {'error': [e.ename, e.evalue, str(e).split('------------------')[1].strip()],
                    'outputs': [_cell_text(c) for c in nb.cells[1:-1] if c.cell_type == 'code']}
        dump = json.loads((work / '__dump__.json').read_text())
    dump['outputs'] = [_cell_text(c) for c in nb.cells[1:-1] if c.cell_type == 'code']
    return dump


def _cell_text(cell) -> str:
    """Text outputs of a cell, with tqdm progress bars dropped."""
    parts = []
    for o in cell.get('outputs', []):
        if o.output_type == 'stream' and o.name == 'stdout':
            parts.append(o.text)
        elif o.output_type == 'execute_result':
            parts.append(o.data.get('text/plain', ''))
    return ''.join(parts)


# Numbers that come out of a scipy optimiser: the Ramberg-Osgood parameters (C or H, n) from ModelSet, the
# Zener-Hollomon regression from curve_fit, statistics over those, and curves predicted from fitted parameters. The
# optimiser's stopping point moves in the 6th significant figure between scipy versions and platforms (the fit error
# itself agrees to ~1e-10), and a std over a tight spread of parameters amplifies that, so they get a looser tolerance.
# Everything else must match to `rtol`.
OPTIMISER_COLUMN = re.compile(r'^(std_|upstd_|downstd_|max_|min_|q1_|q3_)?(C|H|n|lnZ_fit\w*|ZH_fit\w*)$')
OPTIMISER_PATH = re.compile(r'/data/model_id'  # curves predicted from fitted parameters
                            r'|^/datasets/repres_ds/data/repres_id')  # CS2 averages those curves into repres_ds
OPTIMISER_RTOL = 1e-3


def _number(x):
    """Floats stored as strings (ModelSet.fitting_table holds its values as strings) compare as floats."""
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x


def _tex_frame(tex: str) -> dict:
    """A LaTeX table as columns and rows, ignoring the whitespace and escaping that differ between pandas versions."""
    rows = [[_number(c.strip().replace(r'\_', '_').replace('{}', '')) for c in line.rstrip(' \\').split('&')]
            for line in tex.splitlines() if '&' in line]
    return {'columns': [str(c) for c in rows[0]], 'rows': rows[1:]} if rows else {'columns': [], 'rows': []}


def compare(ref, new, path='', rtol=1e-9, diffs=None):
    """Recursively compare two dumps. Floats match when equal to within rtol (OPTIMISER_RTOL for optimiser outputs);
    everything else must be equal."""
    diffs = [] if diffs is None else diffs
    if OPTIMISER_PATH.search(path):
        rtol = max(rtol, OPTIMISER_RTOL)
    if isinstance(ref, str) and isinstance(new, str) and path.endswith('.tex'):
        ref, new = _tex_frame(ref), _tex_frame(new)
    if isinstance(ref, dict) and isinstance(new, dict) and set(ref) == set(new) == {'columns', 'rows'}:
        compare(ref['columns'], new['columns'], f'{path}/columns', rtol, diffs)
        if len(ref['rows']) != len(new['rows']):
            diffs.append((f'{path}/rows', f"length {len(ref['rows'])} != {len(new['rows'])}", None))
        for i, (row_a, row_b) in enumerate(zip(ref['rows'], new['rows'])):
            for col, a, b in zip(ref['columns'], row_a, row_b):
                col_rtol = max(rtol, OPTIMISER_RTOL) if OPTIMISER_COLUMN.match(col) else rtol
                compare(_number(a), _number(b), f'{path}/rows[{i}]/{col}', col_rtol, diffs)
    elif isinstance(ref, dict) and isinstance(new, dict):
        for k in sorted(set(ref) | set(new)):
            if k not in ref or k not in new:
                diffs.append((f'{path}/{k}', 'missing in ' + ('new' if k not in new else 'ref'), None))
            else:
                compare(ref[k], new[k], f'{path}/{k}', rtol, diffs)
    elif isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            diffs.append((path, f'length {len(ref)} != {len(new)}', None))
        for i, (a, b) in enumerate(zip(ref, new)):
            compare(a, b, f'{path}[{i}]', rtol, diffs)
    elif isinstance(ref, float) and isinstance(new, (int, float)) and not isinstance(new, bool):
        if not (math.isclose(ref, new, rel_tol=rtol, abs_tol=1e-12) or (math.isnan(ref) and math.isnan(new))):
            diffs.append((path, f'{ref!r} != {new!r}', abs(ref - new) / max(abs(ref), 1e-300)))
    elif ref != new:
        diffs.append((path, f'{str(ref)[:120]!r} != {str(new)[:120]!r}', None))
    return diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compare', type=Path, help='reference directory to compare against')
    p.add_argument('--only', nargs='*', default=list(NOTEBOOKS))
    p.add_argument('--rtol', type=float, default=1e-9)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for key in args.only:
        print(f'== {key}', flush=True)
        dump = run_notebook(args.examples, NOTEBOOKS[key])
        (args.out / f'{key}.json').write_text(json.dumps(dump, indent=1, sort_keys=True))
        if 'error' in dump:
            print('   FAILED:', *dump['error'])
            failed = True
            continue
        if args.compare:
            ref = json.loads((args.compare / f'{key}.json').read_text())
            diffs = compare({k: v for k, v in ref.items() if k != 'outputs'},
                            {k: v for k, v in dump.items() if k != 'outputs'}, rtol=args.rtol)
            out_diffs = [i for i, (a, b) in enumerate(zip(ref['outputs'], dump['outputs'])) if a != b]
            print(f'   {len(diffs)} value diffs, {len(out_diffs)} printed-output diffs (cells {out_diffs[:10]})')
            for d in diffs[:25]:
                print('  ', *d)
            failed |= bool(diffs)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
