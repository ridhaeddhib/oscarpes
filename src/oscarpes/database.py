"""
oscarpes.database
====================
OSCARDatabase — Lance + Zarr database client for OSCARpes v3.

Replaces the HDF5-based client with a Lance columnar store for fast
metadata queries, ML streaming, and vector search, plus Zarr pools
for scientific pool data.

Usage
-----
::

    db = OSCARDatabase('oscar_db/')
    print(db)                                      # summary table
    e  = db['72aba4d9-...']                        # by UUID
    es = db.find(formula='WSe2')                   # filter → list[OSCAREntry]
    X, y, ids = db.batch_load('intensity_total',
                               formula_contains='WSe2')
    ds = db.as_pytorch_dataset()                   # PyTorch iterable
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .store import (
    lance_path, lance_exists, open_lance,
    lance_filter_one, open_zarr, check_schema_version,
)
from .entry import OSCAREntry

DEFAULT_DB = str(Path.home() / ".oscarpes")


# ── scalar metadata columns (no ARPES arrays, no embedding) ──────────────────

_META_COLS = [
    'entry_id', 'created_at', 'formula', 'dataset_label', 'spc_sha256',
    'crystal_sha', 'geom_sha', 'pot_sha',
    'bravais_type', 'point_group', 'alat_bohr', 'nq', 'nt', 'n_layers',
    'irel', 'nspin', 'fullpot', 'xc_potential', 'fermi_energy_ev', 'scf_status',
    'eminev', 'emaxev',
    'photon_energy_ev', 'polarization', 'theta_inc_deg', 'phi_inc_deg',
    'NK', 'NE', 'adsi',
    'ka', 'k1', 'nk1',
    'stokes_s0', 'stokes_s1_pct', 'stokes_s2_pct', 'stokes_s3_pct',
]


class OSCARDatabase:
    """
    Query interface to an oscarpes v3 Lance + Zarr database.

    Parameters
    ----------
    db_path : str, optional
        Database directory — local path or ``s3://bucket/prefix``.
        Defaults to ``~/.oscarpes/``.

    Examples
    --------
    ::

        db = OSCARDatabase()             # → ~/.oscarpes/
        db = OSCARDatabase('/data/mydb') # custom path
        print(db)
        e  = db['72aba4d9-...']
        es = db.find(formula='WSe2', irel=3)
        X, y, ids = db.batch_load('intensity_total', formula_contains='WSe2')
        df = db.to_dataframe()
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else DEFAULT_DB
        check_schema_version(self.db_path)

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        import numpy as np
        entries = self.list_entries()
        header = f"OSCARDatabase({self.db_path!r}, {len(entries)} entries)"
        if not entries:
            return header
        sep = '  ' + '─' * 60
        lines = [header, sep]
        for e in entries:
            # k range from BZ path
            ka = e.get('ka'); k1v = e.get('k1')
            if ka and k1v:
                ka_arr = np.asarray(list(ka),  dtype=float)
                k1_arr = np.asarray(list(k1v), dtype=float)
                ke_arr = ka_arr + k1_arr
                k1_norm = np.linalg.norm(k1_arr)
                if k1_norm > 0:
                    k1_hat = k1_arr / k1_norm
                    ks = float(np.dot(ka_arr, k1_hat))
                    ke = float(np.dot(ke_arr, k1_hat))
                else:
                    ks = float(np.linalg.norm(ka_arr))
                    ke = float(np.linalg.norm(ke_arr))
                k_str = f'k=[{ks:.3f}, {ke:.3f}] Å⁻¹'
            else:
                k_str = 'k=n/a'
            NK       = e.get('NK', 0); NE = e.get('NE', 0)
            pot_type = 'FullPot' if e.get('fullpot') else 'ASA'
            irel     = e.get('irel', '?')
            nspin    = e.get('nspin', 1)
            mag      = 'magnetic' if nspin == 2 else 'non-magnetic'
            bravais  = e.get('bravais_type') or '?'
            pg       = e.get('point_group')  or '?'
            hv       = e.get('photon_energy_ev', 0.)
            pol      = e.get('polarization', '')
            emax     = e.get('emaxev', float('nan'))
            emin     = e.get('eminev', float('nan'))
            lines += [
                f"  {e['entry_id']}",
                f"    Formula   : {e.get('formula', '?')}",
                f"    Bravais   : {bravais}  point group {pg}",
                f"    Potential : {pot_type}  irel={irel}  {mag}",
                f"    Photon Source      : hν={hv:.1f} eV  pol={pol}",
                f"    ARPES Cross Section: E=[{emax:.2f}, {emin:.2f}] eV  {k_str}  ({NE}×{NK})",
                sep,
            ]
        return '\n'.join(lines)

    # ── core access ───────────────────────────────────────────────────────────

    def __getitem__(self, entry_id: str) -> OSCAREntry:
        row = lance_filter_one(self.db_path, entry_id)
        if row is None:
            raise KeyError(f"Entry {entry_id!r} not found in {self.db_path!r}")
        return OSCAREntry(entry_id, self.db_path, lance_row=row)

    # ── metadata listing ──────────────────────────────────────────────────────

    def list_entries(self) -> List[dict]:
        """Return a list of metadata dicts for all entries (no ARPES arrays)."""
        if not lance_exists(self.db_path):
            return []
        ds  = open_lance(self.db_path)
        tbl = ds.to_table(columns=_META_COLS)
        return tbl.to_pylist()

    # ── filtering ─────────────────────────────────────────────────────────────

    def find(self,
             formula: Optional[str] = None,
             photon_energy_ev: Optional[float] = None,
             photon_energy_tol: float = 2.0,
             photon_energy_ev_min: Optional[float] = None,
             photon_energy_ev_max: Optional[float] = None,
             fermi_energy_ev_min: Optional[float] = None,
             fermi_energy_ev_max: Optional[float] = None,
             eminev_min: Optional[float] = None,
             eminev_max: Optional[float] = None,
             emaxev_min: Optional[float] = None,
             emaxev_max: Optional[float] = None,
             nk_min: Optional[int] = None,
             nk_max: Optional[int] = None,
             polarization: Optional[str] = None,
             irel: Optional[int] = None,
             xc_potential: Optional[str] = None,
             **kwargs) -> List[OSCAREntry]:
        """
        Filter entries by metadata and return matching :class:`OSCAREntry` objects.

        Named parameters
        ----------------
        formula              : str   substring match against formula
        photon_energy_ev     : float exact match within photon_energy_tol
        photon_energy_tol    : float tolerance for photon_energy_ev (default 2 eV)
        photon_energy_ev_min : float lower bound (inclusive) on photon_energy_ev
        photon_energy_ev_max : float upper bound (inclusive) on photon_energy_ev
        fermi_energy_ev_min  : float lower bound on fermi_energy_ev
        fermi_energy_ev_max  : float upper bound on fermi_energy_ev
        eminev_min / _max    : float ARPES energy axis range bounds
        emaxev_min / _max    : float ARPES energy axis range bounds
        nk_min / nk_max      : int   bounds on number of k-points (NK)
        polarization         : str   exact match (e.g. 'LH', 'C+')
        irel                 : int   exact match (1=non-rel, 2=scalar, 3=full-rel)
        xc_potential         : str   exact match (e.g. 'VWN', 'PBE')

        Extra keyword arguments
        -----------------------
        Any scalar column in the Lance schema.  Float values use ±1e-6
        tolerance; all others use exact string/int comparison.
        """
        if not lance_exists(self.db_path):
            return []

        # ── build Lance predicate ─────────────────────────────────────────────
        preds = []
        if formula:
            safe = formula.replace("'", "''")
            preds.append(f"formula LIKE '%{safe}%'")
        if polarization:
            safe = polarization.replace("'", "''")
            preds.append(f"polarization = '{safe}'")
        if irel is not None:
            preds.append(f"irel = {int(irel)}")
        if xc_potential:
            safe = xc_potential.replace("'", "''")
            preds.append(f"xc_potential = '{safe}'")
        if photon_energy_ev is not None:
            lo = photon_energy_ev - photon_energy_tol
            hi = photon_energy_ev + photon_energy_tol
            preds.append(f"photon_energy_ev >= {lo} AND photon_energy_ev <= {hi}")

        # ── range predicates ──────────────────────────────────────────────────
        _range_pairs = [
            ('photon_energy_ev', photon_energy_ev_min, photon_energy_ev_max),
            ('fermi_energy_ev',  fermi_energy_ev_min,  fermi_energy_ev_max),
            ('eminev',           eminev_min,            eminev_max),
            ('emaxev',           emaxev_min,            emaxev_max),
            ('NK',               nk_min,                nk_max),
        ]
        for col, lo, hi in _range_pairs:
            if lo is not None:
                preds.append(f"{col} >= {lo}")
            if hi is not None:
                preds.append(f"{col} <= {hi}")

        # extra kwargs: string/int columns only (floats need special tolerance)
        for key, wanted in kwargs.items():
            if key not in _META_COLS:
                continue
            if isinstance(wanted, float):
                preds.append(
                    f"{key} >= {wanted - 1e-6} AND {key} <= {wanted + 1e-6}"
                )
            elif isinstance(wanted, int):
                preds.append(f"{key} = {wanted}")
            else:
                safe = str(wanted).replace("'", "''")
                preds.append(f"{key} = '{safe}'")

        ds = open_lance(self.db_path)
        filter_str = " AND ".join(preds) if preds else None
        if filter_str:
            tbl = ds.to_table(columns=_META_COLS, filter=filter_str)
        else:
            tbl = ds.to_table(columns=_META_COLS)

        rows = tbl.to_pylist()
        return [OSCAREntry(r['entry_id'], self.db_path) for r in rows]

    # ── CD partner discovery ──────────────────────────────────────────────────

    def find_cd_partner(self, entry: OSCAREntry) -> Optional[OSCAREntry]:
        """
        Find the opposite-helicity partner for circular dichroism computation.

        Searches for an entry with the same ``crystal_sha`` but opposite
        ``polarization`` (C+ ↔ C−, LH ↔ LV).

        Returns
        -------
        OSCAREntry or None if no partner exists.
        """
        if not lance_exists(self.db_path):
            return None

        pol = entry.photon.polarization_label
        opposite = {'C+': 'C-', 'C-': 'C+', 'LH': 'LV', 'LV': 'LH'}.get(pol)
        sha = entry._crys_sha
        if not sha or not opposite:
            return None

        safe_sha = sha.replace("'", "''")
        safe_pol = opposite.replace("'", "''")
        filter_str = (
            f"crystal_sha = '{safe_sha}' "
            f"AND polarization = '{safe_pol}' "
            f"AND entry_id != '{entry.entry_id}'"
        )
        ds  = open_lance(self.db_path)
        tbl = ds.to_table(columns=['entry_id'], filter=filter_str)
        if tbl.num_rows == 0:
            return None
        eid = tbl.column('entry_id')[0].as_py()
        return OSCAREntry(eid, self.db_path)

    # ── batch load ────────────────────────────────────────────────────────────

    def batch_load(self,
                   dataset: str = 'intensity_total',
                   formula_contains: Optional[str] = None,
                   label_attr: str = 'photon_energy_ev',
                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a 2D ARPES array from all matching entries.

        Parameters
        ----------
        dataset : one of 'intensity_total', 'intensity_up', 'intensity_down',
                  'spin_polarization', 'cd_arpes' (computed on the fly as up−down)
        formula_contains : optional formula substring filter
        label_attr : Lance column to use as the y-label (default 'photon_energy_ev')

        Returns
        -------
        X    : np.ndarray  shape (N, NK, NE)
        y    : np.ndarray  shape (N,)
        ids  : list[str]   entry UUIDs
        """
        if not lance_exists(self.db_path):
            return np.empty((0,)), np.array([]), []

        cols = ['entry_id', 'NK', 'NE', label_attr]
        if dataset == 'cd_arpes':
            cols += ['intensity_up', 'intensity_down']
        else:
            cols.append(dataset)

        ds = open_lance(self.db_path)
        if formula_contains:
            safe = formula_contains.replace("'", "''")
            tbl  = ds.to_table(columns=cols,
                                filter=f"formula LIKE '%{safe}%'")
        else:
            tbl = ds.to_table(columns=cols)

        rows = tbl.to_pylist()
        arrs, ys, ids = [], [], []
        for row in rows:
            try:
                NK = int(row['NK'] or 0); NE = int(row['NE'] or 0)
                if NK == 0 or NE == 0:
                    continue
                y_val = float(row.get(label_attr) or 0.)
                if dataset == 'cd_arpes':
                    up   = np.asarray(row['intensity_up'],   dtype=np.float32).reshape(NK, NE)
                    down = np.asarray(row['intensity_down'], dtype=np.float32).reshape(NK, NE)
                    arr  = up - down
                else:
                    arr = np.asarray(row[dataset], dtype=np.float32).reshape(NK, NE)
                arrs.append(arr)
                ys.append(y_val)
                ids.append(row['entry_id'])
            except (ValueError, TypeError, KeyError) as exc:
                import warnings as _w
                _w.warn(
                    f"[batch_load] Skipping entry {row.get('entry_id')!r}: "
                    f"{type(exc).__name__}: {exc}",
                    UserWarning, stacklevel=2,
                )

        X = np.stack(arrs, axis=0) if arrs else np.empty((0,))
        return X, np.array(ys), ids

    # ── pandas ────────────────────────────────────────────────────────────────

    def to_dataframe(self):
        """Return a pandas DataFrame of all entry metadata (no ARPES arrays)."""
        import pandas as pd
        if not lance_exists(self.db_path):
            return pd.DataFrame()
        ds = open_lance(self.db_path)
        return ds.to_table(columns=_META_COLS).to_pandas()

    # ── deletion ──────────────────────────────────────────────────────────────

    def delete(self, entry_ids: List[str], raise_if_missing: bool = False) -> None:
        """
        Delete a list of entries from the Lance dataset.

        Parameters
        ----------
        entry_ids : list of str
            UUIDs of the entries to remove.
        raise_if_missing : bool, default=False
            If True, raise KeyError if any entry_id doesn't exist.

        Raises
        ------
        KeyError
            If raise_if_missing=True and any entry_id doesn't exist.

        Notes
        -----
        This operation removes rows from the ``entries.lance`` table.
        It does NOT remove data from the Zarr pools (crystals, potentials),
        as those might be shared by other entries.
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError("Deleting entries requires lancedb: pip install lancedb")

        if not lance_exists(self.db_path) or not entry_ids:
            return

        # Ensure list (in case generator is passed)
        entry_ids = list(entry_ids)
        if not entry_ids:
            return

        db  = lancedb.connect(self.db_path)
        tbl = db.open_table('entries')

        # Check if entries exist (if requested)
        if raise_if_missing:
            existing_entries = set(tbl.to_pandas()['entry_id'])
            missing_ids = [eid for eid in entry_ids if eid not in existing_entries]
            if missing_ids:
                raise KeyError(f"Entries not found: {missing_ids}")

        # Chunk to avoid huge SQL queries
        chunk_size = 50
        for i in range(0, len(entry_ids), chunk_size):
            chunk = entry_ids[i:i+chunk_size]
            safe_ids = [str(eid).replace("'", "''") for eid in chunk]
            ids_list = ", ".join(f"'{eid}'" for eid in safe_ids)
            tbl.delete(f"entry_id IN ({ids_list})")

        print(f"[OSCAR] Deleted {len(entry_ids)} entries from {self.db_path}")

    # ── PyTorch streaming ─────────────────────────────────────────────────────

    def as_pytorch_dataset(self, filter: Optional[str] = None,
                           batch_size: int = 32,
                           columns: Optional[List[str]] = None):
        """
        Return a Lance PyTorch IterableDataset for ML training.

        Parameters
        ----------
        filter     : optional Lance SQL predicate string
        batch_size : rows per batch
        columns    : columns to include (default: all)

        Returns
        -------
        lance.pytorch.LanceDataset (torch IterableDataset)

        Example
        -------
        ::

            ds = db.as_pytorch_dataset(filter="irel = 3")
            loader = torch.utils.data.DataLoader(ds, batch_size=32)
        """
        try:
            import lancedb
            from lancedb.query import LanceFtsQuery  # noqa: just check lancedb is present
        except ImportError:
            raise ImportError(
                "PyTorch streaming requires lancedb: pip install lancedb"
            )
        try:
            import torch.utils.data as td
        except ImportError:
            raise ImportError("PyTorch streaming requires torch: pip install torch")

        db  = lancedb.connect(self.db_path)
        tbl = db.open_table('entries')

        class _LanceIterDS(td.IterableDataset):
            def __iter__(self_):
                q = tbl.search()
                if filter:
                    q = q.where(filter)
                if columns:
                    q = q.select(columns)
                for batch in q.to_arrow().to_batches(max_chunksize=batch_size):
                    yield {c: batch.column(c).to_pylist() for c in batch.schema.names}

        return _LanceIterDS()

    # ── pool summary ──────────────────────────────────────────────────────────

    def pool_summary(self) -> dict:
        """Return counts of shared pool objects."""
        summary = {
            'entries':        0,
            'crystals':       0,
            'lkkr_geometry':  0,
            'potentials':     0,
        }
        if lance_exists(self.db_path):
            ds = open_lance(self.db_path)
            summary['entries'] = ds.count_rows()

        for pool, subpath in [
            ('crystals',      'crystals.zarr'),
            ('lkkr_geometry', 'lkkr_geometry.zarr'),
            ('potentials',    'potentials.zarr'),
        ]:
            try:
                zg = open_zarr(self.db_path, subpath, mode='r')
                summary[pool] = len(list(zg.keys()))
            except Exception:
                pass

        return summary

    # ── Zarr tree printer ─────────────────────────────────────────────────────

    def print_tree(self, entry_id: Optional[str] = None, max_depth: int = 3):
        """Print the Zarr pool tree for one entry or a summary of all pools."""
        if entry_id:
            row = lance_filter_one(self.db_path, entry_id)
            if row is None:
                print(f'Entry {entry_id!r} not found.')
                return
            pot_sha = row.get('pot_sha') or ''
            crys_sha = row.get('crystal_sha') or ''
            geom_sha = row.get('geom_sha') or ''
            print(f'Entry: {entry_id}  formula={row.get("formula","")}')
            for label, subpath in [
                ('crystal',      f'crystals.zarr/{crys_sha}'),
                ('lkkr_geometry', f'lkkr_geometry.zarr/{geom_sha}'),
                ('potential',    f'potentials.zarr/{pot_sha}'),
            ]:
                if not crys_sha and label == 'crystal':
                    continue
                try:
                    zg = open_zarr(self.db_path, subpath, mode='r')
                    print(f'  [{label}] sha={subpath.split("/")[-1][:12]}…')
                    _print_zarr_tree(zg, indent=4, max_depth=max_depth)
                except Exception:
                    print(f'  [{label}] (not found)')
        else:
            print(self)
            print()
            ps = self.pool_summary()
            print(f"Pool summary: "
                  f"crystals={ps['crystals']}  "
                  f"lkkr_geometry={ps['lkkr_geometry']}  "
                  f"potentials={ps['potentials']}")


def _print_zarr_tree(g, indent: int = 0, max_depth: int = 3, _depth: int = 0):
    """Recursively print a Zarr group tree."""
    if _depth > max_depth:
        return
    import zarr
    for key in sorted(g.keys()):
        item = g[key]
        pad  = ' ' * indent
        if isinstance(item, zarr.Group):
            attrs_brief = dict(list(item.attrs.items())[:2])
            print(f'{pad}📁 {key}/  {attrs_brief}')
            _print_zarr_tree(item, indent + 2, max_depth, _depth + 1)
        else:
            print(f'{pad}  📊 {key}: shape={item.shape} dtype={item.dtype}')
