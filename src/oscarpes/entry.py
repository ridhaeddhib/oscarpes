"""
oscarpes.entry
=================
OSCAREntry — ergonomic wrapper around one v3 database entry.

Data is loaded from two complementary stores:

* **Lance** (``entries.lance``) — metadata scalars + ARPES arrays.
  This is the primary access path; arrays arrive as numpy after the
  PyArrow→numpy conversion.

* **Zarr pools** — scientific pool data keyed by SHA-256:

  - ``crystals.zarr/<sha>/``       → :class:`CrystalData`
  - ``lkkr_geometry.zarr/<sha>/``  → :class:`LKKRGeometryData`
  - ``potentials.zarr/<sha>/``     → :class:`SCFData` + radial / SFN data
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Tuple, List
import warnings
import numpy as np
import zarr

try:
    from ase2sprkkr.sprkkr.spacegroup_info import SpacegroupInfo
except ImportError:
    SpacegroupInfo = None  # type: ignore

from .store import open_zarr, lance_filter_one


# ══════════════════════════════════════════════════════════════════════════════
#  Dataclasses  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrystalData:
    """Bulk crystal identity — loaded from ``crystals.zarr/<sha>/``."""
    alat_bohr: float = 0.
    alat_angstrom: float = 0.
    bravais_type: str = ''
    bravais_index: int = 0
    point_group: str = ''
    nq: int = 0
    nt: int = 0
    type_labels: Optional[List[str]] = None
    # lattice vectors (dimensionless × alat)
    a1: Optional[np.ndarray] = None
    a2: Optional[np.ndarray] = None
    a3: Optional[np.ndarray] = None
    crystal_sha: Optional[str] = None


@dataclass
class LKKRGeometryData:
    """Semi-infinite LKKR layer stack — loaded from ``lkkr_geometry.zarr/<sha>/``."""
    n_layers: int = 0
    alat_2d: float = 0.
    # SPEC_STR parameters
    n_layer: Optional[int] = None
    nlat_g_vec: Optional[int] = None
    n_laydbl: Optional[np.ndarray] = None
    surf_bar: Optional[np.ndarray] = None
    transp_bar: bool = False
    # Layer positions
    layer_z_positions: Optional[np.ndarray] = None
    semiinfinite_positions_bohr: Optional[np.ndarray] = None
    a1_2d: Optional[np.ndarray] = None
    a2_2d: Optional[np.ndarray] = None
    geom_sha: Optional[str] = None
    istr: Optional[np.ndarray] = None
    # Potential barrier (from SPEC.out)
    barrier_ibar:   int             = 0
    barrier_epsx:   float           = 0.
    barrier_zparup: Optional[np.ndarray] = None  # spin-up  [z0, z1, z2]
    barrier_zpardn: Optional[np.ndarray] = None  # spin-down [z0, z1, z2]
    barrier_bparp:  Optional[np.ndarray] = None  # parallel [b0, b1, b2]
    basis_real_2d: Optional[np.ndarray] = None
    basis_recip_2d: Optional[np.ndarray] = None
    # Per-atom semiinfinite data (from in_structur.inp)
    semiinfinite_atype:   Optional[np.ndarray] = None  # (N,) int atom type indices
    semiinfinite_z:       Optional[np.ndarray] = None  # (N,) z in alat units
    semiinfinite_a1_frac: Optional[np.ndarray] = None  # (N,) a1 fractional coords
    semiinfinite_a2_frac: Optional[np.ndarray] = None  # (N,) a2 fractional coords
    # Stacking vectors (one entry per layer, bulk layer repeat)
    stacking_dz_arr:  Optional[np.ndarray] = None  # (nlayer,) z step per layer (alat)
    stacking_da1_arr: Optional[np.ndarray] = None  # (nlayer,) a1 fractional step per layer
    stacking_da2_arr: Optional[np.ndarray] = None  # (nlayer,) a2 fractional step per layer
    # First bulk-repeat layer index (layers[semi_inf_start_layer:] form the repeating unit)
    semi_inf_start_layer: Optional[int] = None


@dataclass
class SCFData:
    # ── from pot file (SCF convergence) ────────────────────────────────────────
    fermi_energy_ev: float = 0.
    fermi_energy_ry: float = 0.
    xc_potential: str = ''
    irel: int = 3
    fullpot: bool = True
    lloyd_pot: bool = True
    scf_iterations: int = 0
    scf_tolerance: float = 0.
    rmsavv: float = 0.
    rmsavb: float = 0.
    scf_status: str = ''
    vmtz_ry: float = 0.
    nktab_pot: int = 0
    ne_energy_mesh: int = 0
    nspin: int = 1
    # ── TAU: k-space integration (from inp file) ────────────────────────────
    bzint: str = 'POINTS'
    nktab: int = 0
    nktab2d: int = 0
    # ── SITES: angular-momentum cutoff ──────────────────────────────────────
    nl: str = ''
    # ── MODE: relativistic/spin ─────────────────────────────────────────────
    lloyd: bool = False
    rel_mode: str = ''
    mdir: Optional[np.ndarray] = None
    # ── CONTROL: sphere construction ────────────────────────────────────────
    krws: int = 1
    krmt: Optional[int] = None
    nonmag: bool = False
    nosym: bool = False
    # ── radial data (from potentials.zarr radial_data/) ─────────────────────
    radial_r: Optional[dict] = None      # {type_label: r_array}
    radial_v_mt: Optional[dict] = None   # {type_label: V_r array}
    radial_rmt: Optional[dict] = None    # {type_label: rmt_bohr}
    radial_rws: Optional[dict] = None    # {type_label: rws_bohr}


@dataclass
class PhotonSourceData:
    """Pure photon / radiation source parameters (SPEC_PH)."""
    photon_energy_ev: float = 0.
    theta_inc_deg: float = 0.
    phi_inc_deg: float = 0.
    polarization_label: str = ''
    stokes_s0: float = 0.
    stokes_s1_pct: float = 0.
    stokes_s2_pct: float = 0.
    stokes_s3_pct: float = 0.
    photon_wavevector: Optional[np.ndarray] = None
    vector_potential_re: Optional[np.ndarray] = None
    vector_potential_im: Optional[np.ndarray] = None


# Backward-compatibility aliases
PhotonData = PhotonSourceData


@dataclass
class ElectronAnalyserData:
    """Photoemission calculation parameters (final-state, detector, energy grid)."""
    theta_el: Optional[np.ndarray] = None
    phi_el: Optional[np.ndarray] = None
    nt_el: Optional[int] = None
    np_el: Optional[int] = None
    spol: int = 4
    pol_e: str = 'PZ'
    typ: int = 1
    pspin: Optional[np.ndarray] = None

@dataclass
class SampleData:
    rotaxis: Optional[np.ndarray] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    ka: Optional[np.ndarray] = None
    k1: Optional[np.ndarray] = None
    k2: Optional[np.ndarray] = None
    k3: Optional[np.ndarray] = None
    k4: Optional[np.ndarray] = None
    nk: Optional[int] = None
    nk2: Optional[int] = None
    nk3: Optional[int] = None
    nk4: Optional[int] = None
    work_function_ev: float = 0.
    imv_initial_ev: float = 0.
    imv_final_ev: float = 0.
    final_state_model: str = ''
    iq_at_surf: Optional[int] = None
    miller_hkl: Optional[np.ndarray] = None
    strver: Optional[int] = None
    del_z_rumpled_bohr: Optional[float] = None
    ne: Optional[int] = None
    eminev: Optional[float] = None
    emaxev: Optional[float] = None
    energy_grid_type: Optional[int] = None


# Backward-compatibility alias
PhotoemissionData = SampleData


@dataclass
class StructureData:
    """Legacy combined structure dataclass — kept for backward compatibility.

    In v3 the data lives in :class:`CrystalData` (bulk lattice) and
    :class:`LKKRGeometryData` (layer stack).  ``OSCAREntry.structure``
    is a property that synthesises this from those two objects.
    """
    alat_bohr: float = 0.
    alat_angstrom: float = 0.
    bravais_type: str = ''
    point_group: str = ''
    spacegroup_info: Optional[object] = field(default=None)
    nq: int = 0
    nt: int = 0
    n_layers: int = 0
    type_labels: Optional[list] = None
    sites_z: Optional[np.ndarray] = None
    semiinfinite_positions_bohr: Optional[np.ndarray] = None
    layer_z_positions: Optional[np.ndarray] = None


# ══════════════════════════════════════════════════════════════════════════════
#  ARPES data wrapper
# ══════════════════════════════════════════════════════════════════════════════

class ARPESData:
    """Wrapper for ARPES arrays loaded from a Lance row dict."""

    def __init__(self, row: dict):
        NK = int(row.get('NK') or 0)
        NE = int(row.get('NE') or 0)
        self.NK = NK
        self.NE = NE

        def _arr1d(key) -> np.ndarray:
            v = row.get(key)
            if v is None:
                return np.array([], dtype=np.float32)
            return np.asarray(v, dtype=np.float32)

        def _arr2d(key) -> np.ndarray:
            v = row.get(key)
            if v is None:
                return np.zeros((NK, NE), dtype=np.float32)
            flat = np.asarray(v, dtype=np.float32)
            if NK > 0 and NE > 0:
                if flat.size != NK * NE:
                    raise ValueError(
                        f"ARPESData: column '{key}' has {flat.size} elements "
                        f"but NK={NK} × NE={NE} = {NK * NE} expected. "
                        "The Lance row may be corrupt or from an incompatible schema version."
                    )
                # ase2sprkkr stores arrays as (NE, NK) in the SPC file.
                # ingest.py assigns NK=spc.shape[0]=NE_actual, NE=spc.shape[1]=NK_actual.
                # flat.reshape(NK, NE) recovers (NE_actual, NK_actual) — correct for imshow.
                return flat.reshape(NK, NE)
            return flat

        self.k_axis            = _arr1d('k_axis')
        self.energy_axis       = _arr1d('energy_axis')
        self.intensity_total   = _arr2d('intensity_total')
        self.intensity_up      = _arr2d('intensity_up')
        self.intensity_down    = _arr2d('intensity_down')
        self.spin_polarization = _arr2d('spin_polarization')

        self.k_parallel_grid = np.array([], dtype=np.float32)
        self.theta_deg       = np.array([], dtype=np.float32)
        self.determinant     = _arr2d('determinant')

    def __repr__(self):
        k0 = self.k_axis[0]  if len(self.k_axis) else float('nan')
        k1 = self.k_axis[-1] if len(self.k_axis) else float('nan')
        e0 = self.energy_axis[0]  if len(self.energy_axis) else float('nan')
        e1 = self.energy_axis[-1] if len(self.energy_axis) else float('nan')
        return (f'ARPESData(NK={self.NK}, NE={self.NE}, '
                f'k=[{k0:.2f}..{k1:.2f}] Å⁻¹, '
                f'E=[{e0:.2f}..{e1:.2f}] eV)')

    @property
    def spin_polarization_masked(self) -> np.ndarray:
        P = self.spin_polarization.copy().astype(float)
        P[np.abs(self.intensity_total) < 1e-10] = np.nan
        return P

    def edc(self, k_val: float) -> Tuple[np.ndarray, np.ndarray]:
        k_idx = np.argmin(np.abs(self.k_axis - k_val))
        return self.energy_axis, self.intensity_total[:, k_idx]

    def mdc(self, e_val: float) -> Tuple[np.ndarray, np.ndarray]:
        e_idx = np.argmin(np.abs(self.energy_axis - e_val))
        return self.k_axis, self.intensity_total[e_idx, :]

    def edc_spin(self, k_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_idx = np.argmin(np.abs(self.k_axis - k_val))
        return self.energy_axis, self.intensity_up[:, k_idx], self.intensity_down[:, k_idx]

    def fermi_map(self, e_tol: float = 0.05) -> np.ndarray:
        mask = np.abs(self.energy_axis) < e_tol
        return self.intensity_total[mask, :].sum(axis=0)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return (self.k_axis[0], self.k_axis[-1],
                self.energy_axis[0], self.energy_axis[-1])


# ══════════════════════════════════════════════════════════════════════════════
#  Zarr pool loaders
# ══════════════════════════════════════════════════════════════════════════════

def _load_crystal(db_path: str, sha: Optional[str]) -> CrystalData:
    """Load CrystalData from ``crystals.zarr/<sha>/``."""
    if not sha:
        return CrystalData(crystal_sha=sha)
    try:
        cg = open_zarr(db_path, f'crystals.zarr/{sha}', mode='r')
    except (zarr.errors.GroupNotFoundError, FileNotFoundError, KeyError, OSError) as exc:
        warnings.warn(
            f"[oscarpes] _load_crystal: cannot open crystals.zarr/{sha!r} — "
            f"{type(exc).__name__}: {exc}. Returning empty CrystalData.",
            UserWarning, stacklevel=2,
        )
        return CrystalData(crystal_sha=sha)

    def _zarr_arr(name):
        return cg[name][:] if name in cg else None

    # type_labels is stored as a JSON list in attrs (zarr v3 compatible)
    tl_attr = cg.attrs.get('type_labels')
    if tl_attr is not None:
        type_labels = list(tl_attr)
    else:
        # fallback: old format stored as zarr array
        raw_tl = _zarr_arr('type_labels')
        type_labels = ([t.decode() if isinstance(t, bytes) else str(t) for t in raw_tl]
                       if raw_tl is not None else [])

    return CrystalData(
        alat_bohr     = float(cg.attrs.get('alat_bohr', 0.)),
        alat_angstrom = float(cg.attrs.get('alat_angstrom', 0.)),
        bravais_type  = str(cg.attrs.get('bravais_type', '')),
        bravais_index = int(cg.attrs.get('bravais_index', 0)),
        point_group   = str(cg.attrs.get('point_group', '')),
        nq            = int(cg.attrs.get('nq', 0)),
        nt            = int(cg.attrs.get('nt', 0)),
        type_labels   = type_labels,
        a1            = _zarr_arr('a1'),
        a2            = _zarr_arr('a2'),
        a3            = _zarr_arr('a3'),
        crystal_sha   = sha,
    )


def _load_lkkr_geom(db_path: str, sha: Optional[str]) -> LKKRGeometryData:
    """Load LKKRGeometryData from ``lkkr_geometry.zarr/<sha>/``."""
    if not sha:
        return LKKRGeometryData(geom_sha=sha)
    try:
        gg = open_zarr(db_path, f'lkkr_geometry.zarr/{sha}', mode='r')
    except (zarr.errors.GroupNotFoundError, FileNotFoundError, KeyError, OSError) as exc:
        warnings.warn(
            f"[oscarpes] _load_lkkr_geom: cannot open lkkr_geometry.zarr/{sha!r} — "
            f"{type(exc).__name__}: {exc}. Returning empty LKKRGeometryData.",
            UserWarning, stacklevel=2,
        )
        return LKKRGeometryData(geom_sha=sha)

    def _zarr_arr(name):
        return gg[name][:] if name in gg else None

    def _iattr(k):
        v = gg.attrs.get(k)
        return int(v) if v is not None else None

    def _fattr(k):
        v = gg.attrs.get(k)
        return float(v) if v is not None else 0.

    return LKKRGeometryData(
        n_layers            = int(gg.attrs.get('n_layers', 0)),
        alat_2d             = float(gg.attrs.get('alat_2d', 0.)),
        n_layer             = _iattr('n_layer'),
        nlat_g_vec          = _iattr('nlat_g_vec'),
        n_laydbl            = _zarr_arr('n_laydbl'),
        surf_bar            = _zarr_arr('surf_bar'),
        transp_bar          = bool(int(gg.attrs.get('transp_bar', 0))),
        layer_z_positions   = _zarr_arr('layer_z_positions'),
        semiinfinite_positions_bohr = _zarr_arr('semiinfinite_positions_bohr'),
        a1_2d               = _zarr_arr('a1_2d'),
        a2_2d               = _zarr_arr('a2_2d'),
        geom_sha            = sha,
        istr                = _zarr_arr('istr'),
        barrier_ibar        = int(gg.attrs.get('barrier_ibar', 0)),
        barrier_epsx        = _fattr('barrier_epsx'),
        barrier_zparup      = _zarr_arr('barrier_zparup'),
        barrier_zpardn      = _zarr_arr('barrier_zpardn'),
        barrier_bparp       = _zarr_arr('barrier_bparp'),
        basis_real_2d       = _zarr_arr('basis_real_2d'),
        basis_recip_2d      = _zarr_arr('basis_recip_2d'),
        semiinfinite_atype   = _zarr_arr('semiinfinite_atype'),
        semiinfinite_z       = _zarr_arr('semiinfinite_z'),
        semiinfinite_a1_frac = _zarr_arr('semiinfinite_a1_frac'),
        semiinfinite_a2_frac = _zarr_arr('semiinfinite_a2_frac'),
        stacking_dz_arr  = _zarr_arr('stacking_dz'),
        stacking_da1_arr = _zarr_arr('stacking_da1'),
        stacking_da2_arr = _zarr_arr('stacking_da2'),
        semi_inf_start_layer = (lambda v: int(v) if v is not None else None)(
            gg.attrs.get('semi_inf_start_layer')),
    )


def _load_radial(pg: 'zarr.Group'):
    """Return (radial_r, radial_v_mt, radial_rmt, radial_rws) dicts from radial_data/potential/."""
    r_dict, v_dict, rmt_dict, rws_dict = {}, {}, {}, {}
    try:
        pot_grp = pg['radial_data']['potential']
        for lbl in pot_grp.keys():
            g = pot_grp[lbl]
            if 'r'   in g: r_dict[lbl]   = g['r'][:]
            if 'V_r' in g: v_dict[lbl]   = g['V_r'][:]
            rmt = g.attrs.get('rmt_bohr')
            rws = g.attrs.get('rws_bohr')
            if rmt is not None: rmt_dict[lbl] = float(rmt)
            if rws is not None: rws_dict[lbl] = float(rws)
    except (KeyError, zarr.errors.GroupNotFoundError, AttributeError, TypeError) as exc:
        warnings.warn(
            f"[oscarpes] _load_radial: could not read radial_data/potential — "
            f"{type(exc).__name__}: {exc}.",
            UserWarning, stacklevel=2,
        )
    return r_dict, v_dict, rmt_dict, rws_dict


def _load_scf(db_path: str, pot_sha: Optional[str]) -> SCFData:
    """Load SCFData from ``potentials.zarr/<sha>/scf/``."""
    if not pot_sha:
        return SCFData()
    try:
        pg = open_zarr(db_path, f'potentials.zarr/{pot_sha}', mode='r')
        sc = pg['scf'] if 'scf' in pg else None
    except (zarr.errors.GroupNotFoundError, FileNotFoundError, KeyError, OSError) as exc:
        warnings.warn(
            f"[oscarpes] _load_scf: cannot open potentials.zarr/{pot_sha!r} — "
            f"{type(exc).__name__}: {exc}. Returning empty SCFData.",
            UserWarning, stacklevel=2,
        )
        return SCFData()

    if sc is None:
        return SCFData()

    _mdir = sc['mdir'][:] if 'mdir' in sc else None

    def _iattr(k, default=0):
        v = sc.attrs.get(k)
        return int(v) if v is not None else default

    def _fattr(k, default=0.):
        v = sc.attrs.get(k)
        return float(v) if v is not None else default

    def _battr(k, default=False):
        v = sc.attrs.get(k)
        return bool(int(v)) if v is not None else default

    rad_r, rad_v, rad_rmt, rad_rws = _load_radial(pg)

    return SCFData(
        fermi_energy_ev = _fattr('fermi_energy_ev'),
        fermi_energy_ry = _fattr('fermi_energy_ry'),
        xc_potential    = str(sc.attrs.get('xc_potential', '')),
        irel            = _iattr('irel', 3),
        fullpot         = _battr('fullpot', True),
        lloyd_pot       = _battr('lloyd_pot', True),
        scf_iterations  = _iattr('scf_iterations'),
        scf_tolerance   = _fattr('scf_tolerance'),
        rmsavv          = _fattr('rmsavv'),
        rmsavb          = _fattr('rmsavb'),
        scf_status      = str(sc.attrs.get('scf_status', '')),
        vmtz_ry         = _fattr('vmtz_ry'),
        nktab_pot       = _iattr('nktab_pot'),
        ne_energy_mesh  = _iattr('ne_energy_mesh'),
        nspin           = _iattr('nspin', 1),
        bzint           = str(sc.attrs.get('bzint', 'POINTS')),
        nktab           = _iattr('nktab'),
        nktab2d         = _iattr('nktab2d'),
        nl              = str(sc.attrs.get('nl', '')),
        lloyd           = _battr('lloyd'),
        rel_mode        = str(sc.attrs.get('rel_mode', '')),
        mdir            = _mdir,
        krws            = _iattr('krws', 1),
        krmt            = int(sc.attrs['krmt']) if 'krmt' in sc.attrs else None,
        nonmag          = _battr('nonmag'),
        nosym           = _battr('nosym'),
        radial_r        = rad_r,
        radial_v_mt     = rad_v,
        radial_rmt      = rad_rmt,
        radial_rws      = rad_rws,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  OSCAREntry
# ══════════════════════════════════════════════════════════════════════════════

class OSCAREntry:
    """
    One SPR-KKR ARPES calculation loaded from an oscarpes v3 database.

    All metadata and ARPES arrays come from the Lance row; scientific pool
    data (crystal, LKKR geometry, SCF) is loaded on construction from the
    Zarr pools referenced by their SHA-256 keys.

    Parameters
    ----------
    entry_id  : str
        UUID of the entry.
    db_path   : str
        Database directory (local path or ``s3://`` URI).
    lance_row : dict, optional
        Pre-fetched Lance row dict.  When None the row is queried from Lance.

    Attributes
    ----------
    entry_id          : str
    formula           : str
    crystal           : CrystalData
    lkkr_geometry     : LKKRGeometryData
    arpes             : ARPESData
    scf               : SCFData
    photon            : PhotonSourceData  (also accessible as photon_source)
    photoemission     : SampleData        (also accessible as sample)
    structure         : StructureData     (legacy combined view — property)
    """

    def __init__(self, entry_id: str, db_path: str,
                 lance_row: Optional[dict] = None):
        self._db_path = db_path

        # ── fetch Lance row ───────────────────────────────────────────────────
        if lance_row is None:
            lance_row = lance_filter_one(db_path, entry_id)
        if lance_row is None:
            raise KeyError(f'Entry {entry_id!r} not found in {db_path!r}')

        row = lance_row
        self.entry_id      = entry_id
        self.formula       = str(row.get('formula') or '')
        self.dataset_label = str(row.get('dataset_label') or '')
        self.created_at    = str(row.get('created_at') or '')

        # SHA references (used by cached_property loaders)
        self._crys_sha = row.get('crystal_sha') or None
        self._geom_sha = row.get('geom_sha')    or None
        self._pot_sha  = row.get('pot_sha')      or None

        # Store raw row so cached_property can build ARPES lazily
        self._lance_row = row

        # ── Photon source (Lance scalars — always eager, tiny footprint) ───────
        def _vec3(key):
            v = row.get(key)
            if v is None:
                return None
            arr = np.asarray(v, dtype=np.float64)
            return arr if arr.any() else None

        def _fget(key, default=0.):
            v = row.get(key)
            return float(v) if v is not None else default

        def _iget(key, default=0):
            v = row.get(key)
            return int(v) if v is not None else default

        self.photon = PhotonSourceData(
            photon_energy_ev    = _fget('photon_energy_ev'),
            theta_inc_deg       = _fget('theta_inc_deg'),
            phi_inc_deg         = _fget('phi_inc_deg'),
            polarization_label  = str(row.get('polarization') or ''),
            stokes_s0           = _fget('stokes_s0'),
            stokes_s1_pct       = _fget('stokes_s1_pct'),
            stokes_s2_pct       = _fget('stokes_s2_pct'),
            stokes_s3_pct       = _fget('stokes_s3_pct'),
            photon_wavevector   = _vec3('photon_wavevector'),
            vector_potential_re = _vec3('jones_vector_re'),
            vector_potential_im = _vec3('jones_vector_im'),
        )

        # ── Sample / photoemission params (Lance scalars) ─────────────────────
        def _vec2(key):
            v = row.get(key)
            if v is None:
                return None
            return np.asarray(list(v), dtype=np.float64)

        self.photoemission = SampleData(
            ne                 = _iget('NE') or None,
            nk                 = _iget('nk1') or _iget('NK') or None,
            ka                 = _vec2('ka'),
            k1                 = _vec2('k1'),
            k2                 = _vec2('k2'),
            nk2                = _iget('nk2') or None,
            k3                 = _vec2('k3'),
            nk3                = _iget('nk3') or None,
            k4                 = _vec2('k4'),
            nk4                = _iget('nk4') or None,
            work_function_ev   = _fget('work_function_ev'),
            imv_initial_ev     = _fget('imv_initial_ev'),
            imv_final_ev       = _fget('imv_final_ev'),
            final_state_model  = str(row.get('final_state_model') or ''),
            iq_at_surf         = _iget('iq_at_surf') or None,
            miller_hkl         = _vec3('miller_hkl'),
            strver             = _iget('strver') or None,
            del_z_rumpled_bohr = _fget('del_z_rumpled_bohr') or None,
            eminev             = _fget('eminev') or None,
            emaxev             = _fget('emaxev') or None,
            energy_grid_type   = _iget('energy_grid_type') or None,
        )

    # ── lazy-loaded pool data (Zarr IO deferred until first access) ──────────

    @cached_property
    def arpes(self) -> 'ARPESData':
        """ARPES arrays from the Lance row — loaded on first access."""
        return ARPESData(self._lance_row)

    @cached_property
    def crystal(self) -> CrystalData:
        """Bulk crystal identity from ``crystals.zarr/<sha>/`` — loaded on first access."""
        return _load_crystal(self._db_path, self._crys_sha)

    @cached_property
    def lkkr_geometry(self) -> LKKRGeometryData:
        """Semi-infinite layer stack from ``lkkr_geometry.zarr/<sha>/`` — loaded on first access."""
        return _load_lkkr_geom(self._db_path, self._geom_sha)

    @cached_property
    def scf(self) -> SCFData:
        """SCF + radial data from ``potentials.zarr/<sha>/`` — loaded on first access."""
        return _load_scf(self._db_path, self._pot_sha)

    # ── backward-compat StructureData property ───────────────────────────────

    @property
    def structure(self) -> StructureData:
        """Legacy combined structure view.

        Synthesised from :attr:`crystal` + :attr:`lkkr_geometry`.
        """
        cr = self.crystal
        gm = self.lkkr_geometry
        return StructureData(
            alat_bohr           = cr.alat_bohr,
            alat_angstrom       = cr.alat_angstrom,
            bravais_type        = cr.bravais_type,
            point_group         = cr.point_group,
            nq                  = cr.nq,
            nt                  = cr.nt,
            n_layers            = gm.n_layers,
            type_labels         = cr.type_labels,
            semiinfinite_positions_bohr = gm.semiinfinite_positions_bohr,
            layer_z_positions   = gm.layer_z_positions,
        )

    # ── backward-compat property aliases ─────────────────────────────────────

    @property
    def photon_source(self) -> PhotonSourceData:
        """Alias for :attr:`photon`."""
        return self.photon

    @property
    def sample(self) -> SampleData:
        """Alias for :attr:`photoemission`."""
        return self.photoemission

    def __repr__(self):
        ph  = self.photon
        sc  = self.scf
        cr  = self.crystal
        a   = self.arpes

        pot_type  = 'FullPot' if sc.fullpot else 'ASA'
        rel       = f'irel={sc.irel}'
        bravais   = cr.bravais_type or '?'
        pg        = cr.point_group  or '?'

        e0 = float(a.energy_axis[0])  if len(a.energy_axis) else float('nan')
        e1 = float(a.energy_axis[-1]) if len(a.energy_axis) else float('nan')

        # Compute k range from BZ path definition in in_structur.inp
        # kA = origin, k1 = translation vector, nk = number of points
        sd = self.photoemission
        _nan = float('nan')
        if sd.ka is not None and sd.k1 is not None and sd.nk is not None:
            ka_arr  = np.asarray(sd.ka, dtype=float)
            k1_arr  = np.asarray(sd.k1, dtype=float)
            ke_arr  = ka_arr + k1_arr
            nk      = int(sd.nk)
            # Use magnitude of start/end k-vectors (Å⁻¹); sign by dot with k1 direction
            k1_norm = np.linalg.norm(k1_arr)
            if k1_norm > 0:
                k1_hat  = k1_arr / k1_norm
                k_start = float(np.dot(ka_arr, k1_hat))
                k_end   = float(np.dot(ke_arr, k1_hat))
            else:
                k_start = float(np.linalg.norm(ka_arr))
                k_end   = float(np.linalg.norm(ke_arr))
            k_npts  = nk
        else:
            k_start = float(a.k_axis[0])  if len(a.k_axis) else _nan
            k_end   = float(a.k_axis[-1]) if len(a.k_axis) else _nan
            k_npts  = a.NK

        lines = [
            f'OSCAREntry  {self.entry_id}',
            f'  Formula   : {self.formula}',
            f'  Bravais   : {bravais}  point group {pg}',
            f'  Potential : {pot_type}  {rel}  {"magnetic" if sc.nspin == 2 else "non-magnetic"}',
            f'  Photon Source      : hν={ph.photon_energy_ev:.1f} eV  pol={ph.polarization_label}',
            f'  ARPES Cross Section: E=[{e0:.2f}, {e1:.2f}] eV  '
            f'k=[{k_start:.3f}, {k_end:.3f}] Å⁻¹  ({a.NE}×{k_npts})',
        ]
        return '\n'.join(lines)

    # ── radial data ───────────────────────────────────────────────────────────

    def _pot_zarr(self, subpath: str = ''):
        """Open a Zarr group inside ``potentials.zarr/<pot_sha>/``."""
        if not self._pot_sha:
            raise ValueError('No pot_sha — potential pool not available for this entry')
        path = f'potentials.zarr/{self._pot_sha}'
        if subpath:
            path = path + '/' + subpath.lstrip('/')
        return open_zarr(self._db_path, path, mode='r')

    def get_radial_potential(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r [Bohr], V(r) [Ry]) for atom type ``label``."""
        g = self._pot_zarr(f'radial_data/potential/{label}')
        r   = np.asarray(g['r'][:]   if 'r'   in g else [])
        V_r = np.asarray(g['V_r'][:] if 'V_r' in g else []).ravel()
        nr  = min(len(r), len(V_r))
        return r[:nr], V_r[:nr]

    def get_radial_charge(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r [Bohr], ρ(r) [e/Bohr³]) for atom type ``label``."""
        g = self._pot_zarr(f'radial_data/charge/{label}')
        r     = np.asarray(g['r'][:]     if 'r'     in g else [])
        rho_r = np.asarray(g['rho_r'][:] if 'rho_r' in g else []).ravel()
        nr    = min(len(r), len(rho_r))
        return r[:nr], rho_r[:nr]

    def get_mesh_info(self) -> list:
        """Return list of mesh attribute dicts from ``radial_data/meshes``."""
        mg = self._pot_zarr('radial_data/meshes')
        # zarr v3 removed .values(); use .group_values() or iterate via keys
        keys = list(mg)
        return [dict(mg[k].attrs) for k in keys]

    def get_raw_inp(self) -> str:
        """Return the raw ``.inp`` file content stored in provenance."""
        pv = self._pot_zarr('provenance')
        # stored as attrs string (zarr v3 compatible)
        val = pv.attrs.get('raw_inp_content')
        if val is not None:
            return str(val)
        # fallback: old format stored as zarr array
        if 'raw_inp_content' in pv:
            v = pv['raw_inp_content'][()]
            return v.decode() if isinstance(v, bytes) else str(v)
        return ''

    def real_atom_labels(self) -> list:
        return [l for l in (self.crystal.type_labels or [])
                if not l.startswith('Vc')]

    def get_source_paths(self) -> dict:
        """Return original file paths stored in provenance."""
        try:
            pv = self._pot_zarr('provenance')
            return {k[:-5]: str(v) for k, v in pv.attrs.items()
                    if k.endswith('_path')}
        except Exception:
            return {}

    # ── ASE Atoms reconstruction ──────────────────────────────────────────────

    def to_ase_atoms(self, semiinfinite: bool = True):
        """Reconstruct an ASE Atoms object from stored structure data."""
        from ase import Atoms
        from ase2sprkkr.sprkkr.sprkkr_atoms import SPRKKRAtoms
        _BOHR_TO_ANG = 0.529177210903

        cr = self.crystal
        gm = self.lkkr_geometry
        tls = cr.type_labels or []

        def _symbol(atype_idx):
            lbl  = tls[atype_idx - 1] if 0 < atype_idx <= len(tls) else ''
            base = lbl.split('_')[0]
            return base if base and not base.startswith('Vc') else None

        if semiinfinite and gm.semiinfinite_positions_bohr is not None and len(gm.semiinfinite_positions_bohr) > 0:
            pos_ang = gm.semiinfinite_positions_bohr * _BOHR_TO_ANG

            # Prefer atomic numbers stored directly (written by current ingest).
            # Fall back to atype → type_label lookup for older databases.
            try:
                gg = open_zarr(self._db_path, f'lkkr_geometry.zarr/{self._geom_sha}', mode='r')
                if 'semiinfinite_atomic_numbers' in gg:
                    from ase.data import chemical_symbols as _chem_sym
                    znums   = gg['semiinfinite_atomic_numbers'][:]
                    symbols = [_chem_sym[int(z)] for z in znums]
                    positions = list(pos_ang)
                else:
                    atypes = gg['semiinfinite_atype'][:] if 'semiinfinite_atype' in gg else np.ones(
                        len(gm.semiinfinite_positions_bohr), dtype=int)
                    symbols, positions = [], []
                    for pos, at in zip(pos_ang, atypes):
                        sym = _symbol(int(at))
                        if sym:
                            symbols.append(sym)
                            positions.append(pos)
            except Exception:
                symbols, positions = [], []
                for pos in pos_ang:
                    symbols.append('X')
                    positions.append(pos)

            if not symbols:
                raise ValueError('No real atoms found in semiinfinite_positions_bohr')

            # Build cell: a1_2d/a2_2d are stored as unit vectors scaled by alat_2d.
            # alat_2d is the 2D in-plane lattice constant (Å).
            alat_2d = float(gm.alat_2d) if gm.alat_2d else 0.
            # Fall back to bulk alat when alat_2d was not stored (old databases)
            if alat_2d <= 0:
                alat_2d = cr.alat_angstrom or 0.
            a1_2d = np.asarray(gm.a1_2d) * alat_2d if gm.a1_2d is not None else None
            a2_2d = np.asarray(gm.a2_2d) * alat_2d if gm.a2_2d is not None else None

            pos_arr    = np.array(positions)
            z_positions = pos_arr[:, 2]
            vacuum = 10.0
            if a1_2d is not None and a2_2d is not None:
                cell = [
                    [a1_2d[0], a1_2d[1], 0.],
                    [a2_2d[0], a2_2d[1], 0.],
                    [0., 0., z_positions.max() - z_positions.min() + vacuum],
                ]
            else:
                # Last resort: orthogonal box spanning atom positions
                span = pos_arr.ptp(axis=0) + vacuum
                cell = np.diag(span)

            atoms = Atoms(symbols=symbols, positions=positions, cell=cell,
                          pbc=[True, True, False])
            SPRKKRAtoms.promote_ase_atoms(atoms)
            return atoms

        else:
            # Bulk from crystals Zarr pool
            _BOHR_TO_ANG = 0.529177210903
            try:
                cg = open_zarr(self._db_path, f'crystals.zarr/{self._crys_sha}', mode='r')
                # Prefer direct pot.atoms data stored by current ingest (mirrors show-structure bulk)
                if 'bulk_positions_bohr' in cg and 'bulk_atomic_numbers' in cg:
                    from ase.data import chemical_symbols as _chem_sym
                    pos     = cg['bulk_positions_bohr'][:] * _BOHR_TO_ANG
                    znums   = cg['bulk_atomic_numbers'][:]
                    symbols = [_chem_sym[int(z)] for z in znums]
                    cell    = cg['bulk_cell_angstrom'][:] if 'bulk_cell_angstrom' in cg else None
                    atoms   = Atoms(symbols=symbols, positions=pos,
                                    cell=cell if cell is not None else np.eye(3),
                                    pbc=True)
                    SPRKKRAtoms.promote_ase_atoms(atoms)
                    return atoms

                # Fallback: reconstruct from scaled site positions
                a1 = cg['a1'][:] if 'a1' in cg else np.array([1., 0., 0.])
                a2 = cg['a2'][:] if 'a2' in cg else np.array([0., 1., 0.])
                a3 = cg['a3'][:] if 'a3' in cg else np.array([0., 0., 1.])
                alat_ang = float(cg.attrs.get('alat_angstrom', 1.))
                sites_g = cg.get('sites') if hasattr(cg, 'get') else None
                if sites_g is None or 'IQ' not in sites_g:
                    raise ValueError('No site data in crystals.zarr')
                IQ = sites_g['IQ'][:]
                xs = sites_g['x'][:]
                ys = sites_g['y'][:]
                zs = sites_g['z'][:]
                iq_to_type = {iq: i+1 for i, iq in enumerate(sorted(set(IQ)))}
            except Exception as exc:
                raise ValueError(f'Cannot reconstruct bulk Atoms: {exc}') from exc

            cell = np.array([a1, a2, a3]) * alat_ang
            symbols, positions = [], []
            for iq, x, y, z in zip(IQ, xs, ys, zs):
                it  = iq_to_type.get(int(iq), 1)
                sym = _symbol(it)
                if sym:
                    symbols.append(sym)
                    pos = (x * np.array(a1) + y * np.array(a2) + z * np.array(a3)) * alat_ang
                    positions.append(pos)

            if not symbols:
                raise ValueError('No real atoms found in bulk site data')

            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            SPRKKRAtoms.promote_ase_atoms(atoms)
            return atoms

    # ── save / copy ───────────────────────────────────────────────────────────

    def save(self, dest_db_path: str) -> None:
        """Copy this entry (Lance row + referenced Zarr pool groups) to another database."""
        import pyarrow as pa
        from .store import lance_append, require_zarr, zarr_exists, init_db

        init_db(dest_db_path)

        # ── copy Lance row ────────────────────────────────────────────────────
        row = lance_filter_one(self._db_path, self.entry_id)
        if row is None:
            raise RuntimeError(f'Entry {self.entry_id!r} disappeared from source database')
        from .store import entries_schema
        schema = entries_schema()
        arrays = []
        for f in schema:
            val = row.get(f.name)
            if isinstance(f.type, pa.lib.ListType):
                arrays.append(pa.array([val], type=f.type))
            else:
                arrays.append(pa.array([val], type=f.type))
        table = pa.table({f.name: arrays[i] for i, f in enumerate(schema)}, schema=schema)
        lance_append(dest_db_path, table)

        # ── copy Zarr pool groups ─────────────────────────────────────────────
        def _copy_zarr(subpath: str):
            if not zarr_exists(self._db_path, subpath):
                return
            try:
                import zarr
                src = open_zarr(self._db_path, subpath, mode='r')
                dst = require_zarr(dest_db_path, subpath)
                zarr.copy(src, dst, if_exists='replace')
            except Exception as exc:
                import warnings
                warnings.warn(f'[save] Could not copy Zarr group {subpath!r}: {exc}')

        if self._crys_sha:
            _copy_zarr(f'crystals.zarr/{self._crys_sha}')
        if self._geom_sha:
            _copy_zarr(f'lkkr_geometry.zarr/{self._geom_sha}')
        if self._pot_sha:
            _copy_zarr(f'potentials.zarr/{self._pot_sha}')

    # ── convenience constructor ───────────────────────────────────────────────

    @classmethod
    def from_directory(cls, calc_dir: str, formula: str = 'unknown',
                       db_path: Optional[str] = None,
                       force: bool = False) -> 'OSCAREntry':
        """Parse a calculation directory and return a loaded OSCAREntry.

        If ``db_path`` is not given, defaults to ``~/.oscarpes/`` (the same
        default used by :func:`~oscarpes.ingest.ingest_directory` and
        :class:`~oscarpes.database.OSCARDatabase`).  Data persists across
        sessions; re-ingestion is skipped automatically via SHA deduplication.
        """
        from .ingest import ingest_directory, DEFAULT_DB
        if db_path is None:
            db_path = DEFAULT_DB
        eid = ingest_directory(calc_dir, db_path, formula=formula, force=force)
        return cls(eid, db_path)
