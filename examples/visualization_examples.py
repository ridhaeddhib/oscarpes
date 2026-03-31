#!/usr/bin/env python3
"""
Small end-to-end demo for the bundled Au ARPES example.

The script bootstraps a local demo database from ``examples/ARPES_K`` when
needed, then writes a few representative figures next to the script.
"""

from __future__ import annotations

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from oscarpes.database import OSCARDatabase
from oscarpes.ingest import ingest_directory
from oscarpes.visualize import arpes_map, arpes_overview, semiinfinite_structure

DB_PATH = os.path.join(HERE, '_demo_db')
CALC_DIR = os.path.join(HERE, 'ARPES_K')


def _ensure_demo_entry():
    db = OSCARDatabase(DB_PATH)
    rows = list(db.list_entries())
    if rows:
        return db[rows[0]['entry_id']]

    print(f'[demo] ingesting bundled example from {CALC_DIR}')
    entry_id = ingest_directory(CALC_DIR, db_path=DB_PATH, formula='Au')
    db = OSCARDatabase(DB_PATH)
    return db[entry_id]


def main():
    entry = _ensure_demo_entry()
    print(f'[demo] entry_id={entry.entry_id} formula={entry.formula}')

    fig = arpes_map(entry)
    fig.savefig(os.path.join(HERE, 'demo_arpes_map.png'), dpi=150, bbox_inches='tight')
    print('[demo] wrote demo_arpes_map.png')

    fig = arpes_overview(entry)
    fig.savefig(os.path.join(HERE, 'demo_arpes_overview.png'), dpi=150, bbox_inches='tight')
    print('[demo] wrote demo_arpes_overview.png')

    fig = semiinfinite_structure(entry, show=False)
    fig.write_html(os.path.join(HERE, 'demo_semiinfinite_structure.html'))
    print('[demo] wrote demo_semiinfinite_structure.html')


if __name__ == "__main__":
    main()
