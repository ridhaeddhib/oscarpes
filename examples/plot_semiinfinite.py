#!/usr/bin/env python3
"""
Load an entry from the OSCAR database and plot the semi-infinite structure.

Usage examples
--------------
    # interactive browser plot; bootstraps a demo database from examples/ARPES_K
    python plot_semiinfinite.py

    # specify entry and database
    python plot_semiinfinite.py --entry osc-Au-1 --db /data/oscar_db

    # save to HTML
    python plot_semiinfinite.py --out structure.html

API (from another script)
--------------------------
    from oscarpes.database import OSCARDatabase
    import oscarpes.visualize as viz

    db = OSCARDatabase('/data/oscar_db')
    e  = db['osc-Au-1']

    fig = viz.semiinfinite_structure(e, show=True)           # 3-D plotly (default)
    fig = viz.semiinfinite_structure(e, use_3d=False)        # 2-D matplotlib
    fig = viz.semiinfinite_structure(e, backend='ase')       # ASE GUI viewer
"""

import argparse
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _default_demo_db(here: str) -> str:
    return os.path.join(here, '_demo_db')


def _default_calc_dir(here: str) -> str:
    return os.path.join(here, 'ARPES_K')


def _ensure_demo_entry(db_path: str, calc_dir: str):
    from oscarpes.database import OSCARDatabase
    from oscarpes.ingest import ingest_directory

    db = OSCARDatabase(db_path)
    rows = list(db.list_entries())
    if rows:
        return db, rows[0]['entry_id']

    if not os.path.isdir(calc_dir):
        raise FileNotFoundError(
            f'No entries found in {db_path!r} and demo calculation directory {calc_dir!r} is missing.'
        )

    print(f'[plot] no entries found, ingesting demo calculation from {calc_dir}')
    entry_id = ingest_directory(calc_dir, db_path=db_path, formula='Au')
    db = OSCARDatabase(db_path)
    return db, entry_id


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db',    default=None,
                        help='Path to OSCAR database (default: examples/_demo_db)')
    parser.add_argument('--calc-dir', default=None,
                        help='Calculation directory used to bootstrap the demo DB (default: examples/ARPES_K)')
    parser.add_argument('--entry', default=None,
                        help='Entry ID to plot (default: first entry in DB)')
    parser.add_argument('--nbulk', type=int,   default=2,
                        help='Bulk repetitions when reading from files (default: 2)')
    parser.add_argument('--vac',   type=float, default=10.0,
                        help='Vacuum height in Å (default: 10)')
    parser.add_argument('--2d',    dest='use_2d', action='store_true',
                        help='2-D matplotlib cross-section instead of 3-D plotly')
    parser.add_argument('--out',   default=None,
                        help='Save plot to file (html for plotly, png/pdf for matplotlib)')
    args = parser.parse_args()

    db_path = args.db or _default_demo_db(HERE)
    if not os.path.isabs(db_path):
        db_path = os.path.join(HERE, db_path)
    calc_dir = args.calc_dir or _default_calc_dir(HERE)
    if not os.path.isabs(calc_dir):
        calc_dir = os.path.join(HERE, calc_dir)

    import oscarpes.visualize as viz

    if args.entry:
        from oscarpes.database import OSCARDatabase
        db = OSCARDatabase(db_path)
        entry_id = args.entry
    else:
        db, entry_id = _ensure_demo_entry(db_path, calc_dir)

    print(f'[plot] loading entry: {entry_id}')
    e = db[entry_id]
    print(f'[plot] formula: {e.formula}')

    fig = viz.semiinfinite_structure(
        e,
        n_bulk=args.nbulk,
        vacuum=args.vac,
        use_3d=not args.use_2d,
        backend='matplotlib' if args.use_2d else 'plotly',
        show=args.out is None,
        filename=args.out,
    )

    if args.out:
        print(f'[plot] saved → {args.out}')
    return fig


if __name__ == '__main__':
    main()
