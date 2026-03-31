"""
oscarpes.store
=================
Low-level storage helpers shared between ingest, entry, and database layers.

Handles both local filesystem and cloud (S3/GCS) paths transparently via
fsspec.  All higher-level modules import from here rather than opening
Lance / Zarr stores directly.

Layout
------
::

    oscar_db/
      entries.lance          ← ML layer  (Lance columnar store)
      crystals.zarr/         ← pool: bulk crystal identity
      lkkr_geometry.zarr/    ← pool: semi-infinite LKKR layer stack
      potentials.zarr/       ← pool: SCF + radial data + shape functions
      nomad/                 ← FAIR layer (NOMAD archive JSONs)
"""
from __future__ import annotations

import datetime
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import zarr

#: Database schema version — bump when the Lance schema or Zarr layout changes.
SCHEMA_VERSION = "3.0.0"


# ── path helpers ──────────────────────────────────────────────────────────────

def _is_remote(db_path: str) -> bool:
    return db_path.startswith(('s3://', 'gs://', 'gcs://', 'abfs://'))


def _join(db_path: str, *parts: str) -> str:
    """Join a db_path (local or remote) with sub-path components."""
    if _is_remote(db_path):
        return db_path.rstrip('/') + '/' + '/'.join(p.strip('/') for p in parts)
    return str(Path(db_path).joinpath(*parts))


def lance_path(db_path: str) -> str:
    """Absolute path (or URI) to the entries Lance dataset."""
    return _join(db_path, 'entries.lance')


def nomad_dir(db_path: str) -> str:
    """Absolute path (or URI) to the NOMAD archive directory."""
    return _join(db_path, 'nomad')


# ── Zarr pool helpers ─────────────────────────────────────────────────────────

def open_zarr(db_path: str, subpath: str, mode: str = 'r') -> zarr.Group:
    """
    Open a Zarr group at ``db_path/subpath``.

    Handles local filesystem and fsspec-backed stores (s3://, gcs://).

    Parameters
    ----------
    db_path : str
        Root database directory (local path or ``s3://bucket/prefix``).
    subpath : str
        Path inside the database, e.g. ``'crystals.zarr/abc123...'``.
    mode : str
        ``'r'`` read-only, ``'r+'`` read-write, ``'w'`` create/overwrite,
        ``'a'`` create or append (default for writes in pool writers).

    Returns
    -------
    zarr.Group
    """
    full = _join(db_path, subpath)
    if _is_remote(full):
        import fsspec
        proto = full.split('://')[0]
        fs    = fsspec.filesystem(proto)
        # zarr v3: FsspecStore replaces the v2 FSStore
        store = zarr.storage.FsspecStore(
            fs,
            read_only=(mode == 'r'),
            path=full.split('://', 1)[1],
        )
        return zarr.open_group(store, mode=mode)
    return zarr.open_group(full, mode=mode)


def require_zarr(db_path: str, subpath: str) -> zarr.Group:
    """Open (or create) a Zarr group in append/create mode."""
    return open_zarr(db_path, subpath, mode='a')


def zarr_exists(db_path: str, subpath: str) -> bool:
    """Return True if the Zarr group exists and has at least one key."""
    try:
        open_zarr(db_path, subpath, mode='r')
        return True
    except (zarr.errors.GroupNotFoundError, FileNotFoundError, KeyError, OSError):
        return False


# ── Lance helpers (via lancedb) ───────────────────────────────────────────────
#
# The raw `lance` PyPI package conflicts with an unrelated package of the same
# name.  We use `lancedb` (which bundles the Apache Lance format) instead.
# `_LanceTable` is a thin adapter that provides the `to_table(columns, filter)`
# interface expected by database.py without requiring the raw lance API.

class _LanceTable:
    """Adapter wrapping a lancedb Table with a lance-dataset-like API."""

    def __init__(self, tbl):
        self._tbl = tbl

    def to_table(self, columns=None, filter: Optional[str] = None):
        """Return a PyArrow Table, optionally restricted by columns and filter."""
        if filter:
            q = self._tbl.search().where(filter)
            if columns:
                q = q.select(columns)
            return q.to_arrow()
        if columns:
            return self._tbl.search().select(columns).to_arrow()
        return self._tbl.to_arrow()

    def count_rows(self, filter: Optional[str] = None) -> int:
        return self._tbl.count_rows(filter=filter) if filter else self._tbl.count_rows()

    def to_pandas(self, columns=None):
        import pandas as pd
        tbl = self.to_table(columns=columns)
        return tbl.to_pandas()


def _ldb_connect(db_path: str):
    import lancedb
    if not _is_remote(db_path):
        Path(db_path).mkdir(parents=True, exist_ok=True)
    return lancedb.connect(db_path)


def lance_exists(db_path: str) -> bool:
    """Return True if the entries table exists in the lancedb store."""
    try:
        db = _ldb_connect(db_path)
        return 'entries' in db.table_names()
    except (ImportError, OSError, FileNotFoundError, RuntimeError):
        return False


def open_lance(db_path: str) -> _LanceTable:
    """Open the entries table (read-only) and return a _LanceTable adapter."""
    db = _ldb_connect(db_path)
    return _LanceTable(db.open_table('entries'))


def lance_filter_one(db_path: str, entry_id: str) -> Optional[dict]:
    """
    Return a single row as a dict, or None if not found.

    Uses lancedb's pushdown predicate — reads only the matching row.
    """
    if not lance_exists(db_path):
        return None
    db  = _ldb_connect(db_path)
    tbl = db.open_table('entries')
    result = tbl.search().where(f"entry_id = '{entry_id}'").to_arrow()
    if result.num_rows == 0:
        return None
    return {col: result.column(col)[0].as_py() for col in result.schema.names}


def lance_append(db_path: str, table):
    """
    Append a PyArrow Table to the entries lancedb table, creating it if needed.

    Parameters
    ----------
    table : pyarrow.Table
        One or more rows conforming to ENTRIES_SCHEMA.
    """
    db = _ldb_connect(db_path)
    try:
        tbl = db.open_table('entries')
        tbl.add(table)
    except Exception:
        # Pass schema explicitly so lancedb doesn't need to infer list sizes
        db.create_table('entries', data=table, schema=entries_schema())


def lance_upsert(db_path: str, table, on: str = 'entry_id'):
    """Upsert rows into the entries table, matching on ``on``."""
    db = _ldb_connect(db_path)
    try:
        tbl = db.open_table('entries')
    except Exception:
        db.create_table('entries', data=table, schema=entries_schema())
        return

    tbl.merge_insert(on=on) \
       .when_matched_update_all() \
       .when_not_matched_insert_all() \
       .execute(table)


def lance_has_spc(db_path: str, spc_sha256: str) -> bool:
    """Return True if a row with this spc_sha256 already exists (duplicate check)."""
    if not lance_exists(db_path):
        return False
    db  = _ldb_connect(db_path)
    tbl = db.open_table('entries')
    result = tbl.search().where(
        f"spc_sha256 = '{spc_sha256}'"
    ).select(['entry_id']).to_arrow()
    return result.num_rows > 0


def lance_find_entry_id(db_path: str, spc_sha256: str) -> Optional[str]:
    """Return existing entry_id for a given spc_sha256, or None."""
    if not lance_exists(db_path):
        return None
    db  = _ldb_connect(db_path)
    tbl = db.open_table('entries')
    result = tbl.search().where(
        f"spc_sha256 = '{spc_sha256}'"
    ).select(['entry_id']).to_arrow()
    if result.num_rows == 0:
        return None
    return result.column('entry_id')[0].as_py()


# ── entry ID counter (Materials-Project-style: osc-Au-1, osc-WSe2-2, …) ──────

def _counter_path(db_path: str) -> Path:
    return Path(_join(db_path, 'entry_ids.json'))


def alloc_entry_id(db_path: str, formula: str) -> str:
    """
    Allocate the next ``osc-<formula>-<n>`` ID for a given formula.

    The counter is derived from the highest numeric suffix already present in
    the Lance table for this formula, so it stays consistent after deletions.
    A fallback ``entry_ids.json`` counter is used when Lance is not yet
    initialised.

    Examples
    --------
    >>> alloc_entry_id('oscar_db/', 'Au')
    'osc-Au-1'
    >>> alloc_entry_id('oscar_db/', 'WSe2')
    'osc-WSe2-1'
    """
    prefix = f'osc-{formula}-'

    # ── derive max from existing Lance entries (survives deletes) ─────────────
    max_n = 0
    lance_ok = False
    try:
        if lance_exists(db_path):
            import lancedb
            tbl = lancedb.connect(_join(db_path)).open_table('entries')
            ids = tbl.search().select(['entry_id']).to_arrow() \
                     .column('entry_id').to_pylist()
            lance_ok = True
            for eid in ids:
                if isinstance(eid, str) and eid.startswith(prefix):
                    try:
                        max_n = max(max_n, int(eid[len(prefix):]))
                    except ValueError:
                        pass
    except (ImportError, OSError, RuntimeError, AttributeError):
        pass

    # ── fallback: use persisted counter only when Lance is unavailable ────────
    if not lance_ok:
        p = _counter_path(db_path)
        counters: dict = json.loads(p.read_text()) if p.exists() else {}
        max_n = counters.get(formula, 0)

    n = max_n + 1

    # keep the file counter in sync with actual Lance state
    p = _counter_path(db_path)
    counters: dict = json.loads(p.read_text()) if p.exists() else {}
    counters[formula] = max_n  # write current max, not the new n
    p.write_text(json.dumps(counters, indent=2))

    return f'osc-{formula}-{n}'


# ── database initialisation ───────────────────────────────────────────────────

_POOL_NAMES = ('crystals.zarr', 'lkkr_geometry.zarr', 'potentials.zarr', 'entries.zarr')


def init_db(db_path: str) -> None:
    """
    Ensure the database directory structure exists.

    Creates the root Zarr groups for every pool so that ``open_zarr(pool, mode='r')``
    works even before any data has been written.  Also writes ``metadata.json``
    with the current schema version on first call.
    """
    if not _is_remote(db_path):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        Path(_join(db_path, 'nomad')).mkdir(exist_ok=True)
        for pool in _POOL_NAMES:
            zarr.open_group(_join(db_path, pool), mode='a')
        meta_path = Path(_join(db_path, 'metadata.json'))
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                'schema_version': SCHEMA_VERSION,
                'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, indent=2))


def check_schema_version(db_path: str) -> None:
    """
    Warn if the database schema version does not match the current code.

    Called automatically by :class:`~oscarpes.database.OSCARDatabase` on open.
    Safe to call on remote paths — skipped silently.
    """
    if _is_remote(db_path):
        return
    meta_path = Path(_join(db_path, 'metadata.json'))
    if not meta_path.exists():
        warnings.warn(
            f"[oscarpes] {db_path!r} has no metadata.json — "
            "database may have been created by an older version. "
            "Run store.migrate_entries_schema() to upgrade.",
            UserWarning, stacklevel=3,
        )
        return
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    stored = meta.get('schema_version', 'unknown')
    if stored != SCHEMA_VERSION:
        warnings.warn(
            f"[oscarpes] Schema version mismatch: database has {stored!r}, "
            f"current code expects {SCHEMA_VERSION!r}. "
            "Call store.migrate_entries_schema() or re-ingest.",
            UserWarning, stacklevel=3,
        )


# ── PyArrow schema ────────────────────────────────────────────────────────────

def entries_schema():
    """Return the PyArrow schema for the entries Lance dataset."""
    import pyarrow as pa
    return pa.schema([
        # Identity
        pa.field('entry_id',          pa.string()),
        pa.field('created_at',        pa.string()),   # ISO-8601 string
        pa.field('formula',           pa.string()),
        pa.field('dataset_label',     pa.string()),
        pa.field('spc_sha256',        pa.string()),

        # Pool SHA references
        pa.field('crystal_sha',       pa.string()),
        pa.field('geom_sha',          pa.string()),
        pa.field('pot_sha',           pa.string()),

        # Crystal (denormalized for fast filter — no Zarr join needed)
        pa.field('bravais_type',      pa.string()),
        pa.field('point_group',       pa.string()),
        pa.field('alat_bohr',         pa.float64()),
        pa.field('nq',                pa.int32()),
        pa.field('nt',                pa.int32()),
        pa.field('n_layers',          pa.int32()),

        # SCF
        pa.field('irel',              pa.int8()),
        pa.field('nspin',             pa.int8()),
        pa.field('fullpot',           pa.bool_()),
        pa.field('xc_potential',      pa.string()),
        pa.field('fermi_energy_ev',   pa.float64()),
        pa.field('scf_status',        pa.string()),
        # ARPES E range (from SPC energy axis)
        pa.field('eminev',            pa.float64()),
        pa.field('emaxev',            pa.float64()),

        # Photoemission
        pa.field('photon_energy_ev',  pa.float64()),
        pa.field('polarization',      pa.string()),
        pa.field('theta_inc_deg',     pa.float64()),
        pa.field('phi_inc_deg',       pa.float64()),
        pa.field('NK',                pa.int32()),
        pa.field('NE',                pa.int32()),
        pa.field('adsi',              pa.string()),
        # k-path definition from .inp (BZ scan: kA origin, k1..k4 translations, nk1..nk4 points)
        pa.field('ka',                pa.list_(pa.float64())),
        pa.field('k1',                pa.list_(pa.float64())),
        pa.field('nk1',               pa.int32()),
        pa.field('k2',                pa.list_(pa.float64())),
        pa.field('nk2',               pa.int32()),
        pa.field('k3',                pa.list_(pa.float64())),
        pa.field('nk3',               pa.int32()),
        pa.field('k4',                pa.list_(pa.float64())),
        pa.field('nk4',               pa.int32()),
        # Stokes vector (from SPEC.out)
        pa.field('stokes_s0',         pa.float64()),
        pa.field('stokes_s1_pct',     pa.float64()),
        pa.field('stokes_s2_pct',     pa.float64()),
        pa.field('stokes_s3_pct',     pa.float64()),
        # Photon geometry vectors (from SPEC.out) — shape (3,) each
        pa.field('photon_wavevector',  pa.list_(pa.float64())),
        pa.field('jones_vector_re',    pa.list_(pa.float64())),
        pa.field('jones_vector_im',    pa.list_(pa.float64())),
        # Potential barrier (SPEC.out) — semi-infinite surface boundary
        pa.field('barrier_ibar',       pa.int32()),
        pa.field('barrier_epsx',       pa.float64()),
        pa.field('barrier_zparup',     pa.list_(pa.float64())),
        pa.field('barrier_zpardn',     pa.list_(pa.float64())),
        pa.field('barrier_bparp',      pa.list_(pa.float64())),

        # Minimal promoted runtime metadata; rich parsed payloads live in Zarr JSON sidecars.
        pa.field('kkrspec_version',          pa.string()),
        pa.field('execution_datetime',       pa.string()),
        pa.field('mpi_nprocs',               pa.int32()),
        pa.field('cpu_time_sec',             pa.float64()),
        pa.field('wall_time_sec',            pa.float64()),
        pa.field('stop_status',              pa.string()),
        pa.field('slurm_partition',          pa.string()),
        pa.field('slurm_ntasks',             pa.int32()),

        # ARPES arrays stored as flat float32 lists (stored row-major from (NK,NE),
        # loaded as (NE, NK): energy axis-0, k∥ axis-1)
        pa.field('k_axis',            pa.list_(pa.float32())),
        pa.field('energy_axis',       pa.list_(pa.float32())),
        pa.field('intensity_total',   pa.list_(pa.float32())),
        pa.field('intensity_up',      pa.list_(pa.float32())),
        pa.field('intensity_down',    pa.list_(pa.float32())),
        pa.field('spin_polarization', pa.list_(pa.float32())),
        pa.field('determinant',       pa.list_(pa.float32())),
    ])


def migrate_entries_schema(db_path: str) -> int:
    """
    Add any missing columns from :func:`entries_schema` to an existing table.

    Returns the number of newly added columns.
    """
    import pyarrow as pa

    if not lance_exists(db_path):
        return 0

    db = _ldb_connect(db_path)
    tbl = db.open_table('entries')
    current_names = set(tbl.schema.names)
    missing = [field for field in entries_schema() if field.name not in current_names]
    if not missing:
        return 0

    tbl.add_columns(pa.schema(missing))
    return len(missing)
