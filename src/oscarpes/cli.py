"""
oscarpes.cli
============
Command-line interface for OSCARpes.

Usage
-----
::

    oscarpes check-db [--db-path PATH] [--fix] [--json]

Sub-commands
------------
check-db
    Validate database integrity: verify that every Lance entry has its
    Zarr pool objects present and complete, check schema version, and
    report orphaned pool groups.  Exit code 0 = OK, 1 = warnings, 2 = errors.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List


def _cmd_check_db(args) -> int:
    """Validate database integrity. Returns exit code (0/1/2)."""
    from .store import (
        lance_exists, open_lance, zarr_exists, open_zarr,
        check_schema_version, migrate_entries_schema, SCHEMA_VERSION,
    )
    from .database import _META_COLS
    from .ingest import _pot_zarr_complete, DEFAULT_DB

    db_path = args.db_path or DEFAULT_DB

    results: dict = {
        'db_path': db_path,
        'schema_version_expected': SCHEMA_VERSION,
        'schema_warnings': [],
        'entries': [],
        'summary': {},
    }

    # ── schema version ────────────────────────────────────────────────────────
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        check_schema_version(db_path)
    for w in caught:
        results['schema_warnings'].append(str(w.message))

    # ── apply fixes if requested ──────────────────────────────────────────────
    if args.fix:
        n = migrate_entries_schema(db_path)
        if n:
            results['schema_warnings'].append(
                f'migrate_entries_schema added {n} column(s).'
            )

    # ── check Lance table ────────────────────────────────────────────────────
    if not lance_exists(db_path):
        msg = f'No Lance entries table found at {db_path!r}.'
        results['summary'] = {'status': 'ERROR', 'message': msg}
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f'[check-db] ERROR: {msg}', file=sys.stderr)
        return 2

    ds   = open_lance(db_path)
    rows = ds.to_table(columns=[
        'entry_id', 'crystal_sha', 'geom_sha', 'pot_sha',
    ]).to_pylist()

    pass_count = warn_count = fail_count = 0

    for row in rows:
        eid  = row.get('entry_id', '?')
        crys = row.get('crystal_sha') or None
        geom = row.get('geom_sha')    or None
        pot  = row.get('pot_sha')     or None

        issues: List[str] = []

        if not crys:
            issues.append('crystal_sha is NULL')
        elif not zarr_exists(db_path, f'crystals.zarr/{crys}'):
            issues.append(f'crystals.zarr/{crys[:12]}… missing')

        if geom and not zarr_exists(db_path, f'lkkr_geometry.zarr/{geom}'):
            issues.append(f'lkkr_geometry.zarr/{geom[:12]}… missing')

        if not pot:
            issues.append('pot_sha is NULL')
        elif not _pot_zarr_complete(db_path, pot):
            issues.append(f'potentials.zarr/{pot[:12]}… incomplete')

        if issues:
            status = 'FAIL'
            fail_count += 1
        else:
            status = 'PASS'
            pass_count += 1

        results['entries'].append({
            'entry_id': eid,
            'status':   status,
            'issues':   issues,
        })

    # ── orphan detection: pool keys vs unique SHAs in Lance ──────────────────
    orphans: dict = {}
    lance_sha_sets = {
        'crystals.zarr':      set(r.get('crystal_sha') or '' for r in rows) - {''},
        'lkkr_geometry.zarr': set(r.get('geom_sha')    or '' for r in rows) - {''},
        'potentials.zarr':    set(r.get('pot_sha')      or '' for r in rows) - {''},
    }
    for pool, referenced in lance_sha_sets.items():
        try:
            zg = open_zarr(db_path, pool, mode='r')
            pool_keys = set(zg.keys())
        except Exception:
            pool_keys = set()
        orphan_keys = pool_keys - referenced
        if orphan_keys:
            orphans[pool] = sorted(orphan_keys)
            warn_count += 1

    results['summary'] = {
        'total':     len(rows),
        'pass':      pass_count,
        'fail':      fail_count,
        'warnings':  warn_count,
        'orphans':   orphans,
    }

    if not args.json:
        _print_check_report(results)

    else:
        print(json.dumps(results, indent=2))

    if fail_count:
        return 2
    if warn_count or results['schema_warnings']:
        return 1
    return 0


def _print_check_report(results: dict) -> None:
    """Print a human-readable integrity report."""
    db   = results['db_path']
    summ = results['summary']
    print(f'\noscarpes check-db  →  {db}')
    print('─' * 60)

    for w in results.get('schema_warnings', []):
        print(f'  WARN  {w}')

    for e in results['entries']:
        mark = '✓' if e['status'] == 'PASS' else '✗'
        print(f'  {mark}  {e["entry_id"]}')
        for iss in e['issues']:
            print(f'       ↳ {iss}')

    if summ:
        print('─' * 60)
        print(f'  Total: {summ["total"]}  |  '
              f'Pass: {summ["pass"]}  |  Fail: {summ["fail"]}  |  '
              f'Warnings: {summ["warnings"]}')
        for pool, keys in summ.get('orphans', {}).items():
            print(f'  Orphaned in {pool}: {len(keys)} group(s)')
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='oscarpes',
        description='OSCARpes database tools',
    )
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')

    p_check = sub.add_parser(
        'check-db',
        help='Validate database integrity (Zarr pools, schema version, orphans)',
    )
    p_check.add_argument(
        '--db-path', default=None,
        help='Database directory (default: ~/.oscarpes/)',
    )
    p_check.add_argument(
        '--fix', action='store_true',
        help='Run migrate_entries_schema() to add missing columns',
    )
    p_check.add_argument(
        '--json', action='store_true',
        help='Output results as JSON to stdout',
    )

    args = parser.parse_args()

    if args.command == 'check-db':
        sys.exit(_cmd_check_db(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
