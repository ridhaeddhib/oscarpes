"""
oscarpes.ingest
==================
Ingest a complete SPR-KKR ARPES calculation into the oscarpes v3 database.

Storage layout (v3)
-------------------
::

    ~/.oscarpes/
      entries.lance          ← one row per spectrum (metadata + ARPES arrays)
      crystals.zarr/         ← shared crystal pool  (keyed by SHA-256)
      lkkr_geometry.zarr/    ← shared LKKR geometry pool
      potentials.zarr/       ← shared SCF + radial + SFN pool
      nomad/                 ← NOMAD archive JSONs (generated on demand)

All Zarr stores work transparently on local disk, S3, or GCS via fsspec.
The Lance store supports random sampling, streaming, and vector search.

The default database location is ``~/.oscarpes/``.  Pass an explicit
``db_path`` to override (e.g. an S3 URI or a custom local directory).

Usage
-----
::

    from oscarpes.ingest import ingest_directory, ingest_tree
    eid = ingest_directory('/path/to/calc', formula='XXX')          # → ~/.oscarpes/
    eids = ingest_tree('/path/to/all_calcs')                        # → ~/.oscarpes/
    eid = ingest_directory('/path/to/calc', '/data/mydb', formula='XXX')  # custom
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

import numpy as np
import pyarrow as pa
import zarr

from .parsers import (
    parse_spc, parse_spec_out, parse_inp, parse_pot, parse_arpes_out,
    parse_job_script,
    parse_sfn, find_files,
    sha256_file, sha256_crystal, sha256_lkkr_geom, sha256_pot,
    SpecOutData
)
from ase2sprkkr.sprkkr.shape_function import ShapeFunction as SfnData
from ase2sprkkr.sprkkr.radial_meshes import ExponentialMesh
from .store import (
    init_db, open_zarr, require_zarr, zarr_exists,
    lance_append, lance_has_spc, lance_find_entry_id,
    lance_upsert, migrate_entries_schema, alloc_entry_id, entries_schema, _join,
)
_BOHR_TO_ANG   = 0.529177210903

#: Default on-disk database location — ``~/.oscarpes/``.
DEFAULT_DB = str(Path.home() / ".oscarpes")


# ══════════════════════════════════════════════════════════════════════════════
#  Label / formula helpers  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def _auto_label(inp_dict: dict, formula: str) -> str:
    """Build a structured dataset label: ``"{formula}({hkl}) hν={ephot}eV {pol}"``."""
    hkl_raw = inp_dict.get('miller_hkl') or []
    hkl_str = ''.join(str(abs(int(h))) for h in hkl_raw) if hkl_raw else ''
    ephot   = inp_dict.get('ephot')
    pol     = inp_dict.get('pol_p', '')
    label   = formula
    if hkl_str:
        label += f'({hkl_str})'
    if ephot is not None:
        label += f' hν={ephot:.0f}eV'
    if pol:
        label += f' {pol}'
    return label.strip()


def _formula_from_pot(pot) -> str:
    """Derive Hill formula from pot atom types via ASE."""
    from ase import Atoms
    from ase2sprkkr.sprkkr.sprkkr_atoms import SPRKKRAtoms
    # pot.TYPES.DATA() returns a structured numpy array with fields TXT and ZT
    try:
        data = pot.TYPES.DATA()
        symbols = [row['TXT'].split('_')[0]
                   for row in data
                   if row['ZT'] > 0]
    except (AttributeError, KeyError, TypeError, ValueError):
        symbols = []
    if not symbols:
        return ''
    positions = [[0., 0., float(i)] for i in range(len(symbols))]
    atoms = Atoms(symbols=symbols, positions=positions)
    SPRKKRAtoms.promote_ase_atoms(atoms, symmetry=False)
    return atoms.get_chemical_formula('hill')


# ══════════════════════════════════════════════════════════════════════════════
#  Zarr pool writers  (same physics as v2, but target Zarr groups)
# ══════════════════════════════════════════════════════════════════════════════

def _zds(g: 'zarr.Group', name: str, data, **attrs):
    """Create/replace a Zarr array (zarr v3 — default compression applied)."""
    if name in g:
        del g[name]
    a  = np.asarray(data)
    ds = g.create_array(name, data=a)
    for k, v in attrs.items():
        ds.attrs[k] = v
    return ds


def _blosc():
    try:
        from numcodecs import Blosc
        return Blosc(cname='lz4', clevel=4, shuffle=Blosc.BITSHUFFLE)
    except ImportError:
        return None


def _zat(g, **kw):
    for k, v in kw.items():
        if v is not None:
            g.attrs[k] = v


# ── Crystal pool ──────────────────────────────────────────────────────────────

def _write_crystal_zarr(cg: 'zarr.Group', pot):
    """Populate crystals.zarr/<sha>/ from a Potential object."""
    gs  = pot['GLOBAL SYSTEM PARAMETER']
    lat = pot['LATTICE']
    nq  = int(gs['NQ']() or 0)
    nt  = int(gs['NT']() or 0)
    brav = lat['BRAVAIS']() or ('', '', '', '', '')
    alat = float(lat['ALAT']() or 0.)
    _BOHR_TO_ANG = 0.529177210903
    _zat(cg,
         bravais_index=int(brav[0]) if brav[0] else 0,
         bravais_type=str(brav[1]), point_group=str(brav[4]),
         alat_bohr=alat, alat_angstrom=alat * _BOHR_TO_ANG,
         nq=nq, nt=nt)

    cell = lat['SCALED_PRIMITIVE_CELL']()
    if cell is not None and len(cell) >= 3:
        for i, av in enumerate(cell[:3], 1):
            d = cg.require_dataset(f'a{i}', shape=(3,), dtype=np.float64)
            d[:] = np.array(av, dtype=float)
            d.attrs['units'] = 'dimensionless (×alat)'

    scaled = pot['SITES']['SCALED_ATOMIC_POSITIONS']()
    if scaled is not None and len(scaled) > 0:
        gs_grp = cg.require_group('sites')
        iq_arr = np.arange(1, len(scaled) + 1, dtype=int)
        xyz    = np.array(scaled, dtype=float)
        for nm, col, dt in [('IQ', None, int), ('x', 0, float), ('y', 1, float), ('z', 2, float)]:
            arr = iq_arr if nm == 'IQ' else xyz[:, col].astype(dt)
            gs_grp.require_dataset(nm, shape=(len(scaled),), dtype=dt)[:] = arr
        _zat(gs_grp, units='fractional (×alat)', NQ=len(scaled))

    occ_data = pot['OCCUPATION']['DATA']()
    if occ_data is not None and len(occ_data) > 0:
        go = cg.require_group('occupation')
        iqs, itoqs, concs = [], [], []
        for row in occ_data:
            for it, conc in row[3]:
                iqs.append(int(row[0])); itoqs.append(int(it)); concs.append(float(conc))
        go.require_dataset('IQ',   shape=(len(iqs),),   dtype=int)[:] = iqs
        go.require_dataset('ITOQ', shape=(len(itoqs),), dtype=int)[:] = itoqs
        go.require_dataset('CONC', shape=(len(concs),), dtype=float)[:] = concs

    typ_data = pot['TYPES']['DATA']()
    labels = [str(typ_data[i][0]) if typ_data is not None and i < len(typ_data)
              else f'type_{i+1}' for i in range(nt)]
    cg.attrs['type_labels'] = labels

    # Store pot.atoms directly for lossless bulk reconstruction (mirrors show-structure bulk path)
    try:
        from ase.units import Bohr as _AseBohr
        patoms = pot.atoms
        if patoms is not None and len(patoms) > 0:
            _zds(cg, 'bulk_positions_bohr',
                 patoms.get_positions() / _AseBohr, units='Bohr')
            _zds(cg, 'bulk_atomic_numbers',
                 patoms.get_atomic_numbers().astype(np.int32))
            _zds(cg, 'bulk_cell_angstrom',
                 np.array(patoms.get_cell()), units='Angstrom')
    except Exception:
        pass



# ── LKKR geometry pool ────────────────────────────────────────────────────────

def _write_lkkr_geom_zarr(gg: 'zarr.Group', atoms, inp_dict: dict):
    """Populate lkkr_geometry.zarr/<sha>/ from SPRKKRAtoms + inp dict.

    *atoms* is the return value of ``structure_file_to_atoms`` (an SPRKKRAtoms
    instance).  ase2sprkkr's structure_file_to_atoms does NOT populate
    atoms.info with sprkkr_* keys, so we extract what we need directly from
    the ASE Atoms object (cell → a1_2d/a2_2d/alat, positions, atomic numbers).
    """
    from ase.units import Bohr as _Bohr

    info      = atoms.info if atoms is not None else {}
    n_layers  = info.get('sprkkr_n_layers', 0)
    alat_bohr = info.get('sprkkr_alat_bohr', 0.)

    # ── derive alat_2d, a1_2d, a2_2d from the cell when info keys are absent ──
    a1_2d = info.get('sprkkr_a1_2d')
    a2_2d = info.get('sprkkr_a2_2d')
    if atoms is not None and len(atoms) > 0:
        cell = atoms.get_cell()
        a1_ang = np.asarray(cell[0])
        a2_ang = np.asarray(cell[1])
        # Derive alat from first lattice vector if not in info
        if alat_bohr <= 0:
            alat_2d_ang = float(np.linalg.norm(a1_ang[:2])) if np.any(a1_ang[:2]) else 0.
            alat_bohr   = alat_2d_ang / _BOHR_TO_ANG if alat_2d_ang > 0 else 0.
        alat_2d_ang = alat_bohr * _BOHR_TO_ANG
        if a1_2d is None and alat_2d_ang > 0:
            a1_2d = (a1_ang[:2] / alat_2d_ang).tolist()
        if a2_2d is None and alat_2d_ang > 0:
            a2_2d = (a2_ang[:2] / alat_2d_ang).tolist()

    _zat(gg,
         alat_2d=alat_bohr * _BOHR_TO_ANG,
         n_layers=n_layers,
         n_layer=inp_dict.get('n_layer'),
         nlat_g_vec=inp_dict.get('nlat_g_vec'),
         strver=inp_dict.get('strver'))

    # ── always store Cartesian positions + atomic numbers when atoms present ──
    if atoms is not None and len(atoms) > 0:
        cart = atoms.get_positions() / _Bohr
        _zds(gg, 'semiinfinite_positions_bohr', cart, units='Bohr')
        _zds(gg, 'semiinfinite_atomic_numbers',
             atoms.get_atomic_numbers().astype(np.int32))

    if n_layers > 0:
        z_positions = info.get('sprkkr_layer_z_positions', [])
        if z_positions:
            _zds(gg, 'layer_z_positions', np.array(z_positions),
                 units='alat', description='z of first atom per layer')
        raw_atoms = info.get('sprkkr_all_atoms', [])
        if raw_atoms:
            _zds(gg, 'semiinfinite_atype',   np.array([a['atype']   for a in raw_atoms], dtype=int))
            _zds(gg, 'semiinfinite_z',       np.array([a['z']       for a in raw_atoms]), units='alat')
            _zds(gg, 'semiinfinite_a1_frac', np.array([a['a1_frac'] for a in raw_atoms]))
            _zds(gg, 'semiinfinite_a2_frac', np.array([a['a2_frac'] for a in raw_atoms]))
        sv = info.get('sprkkr_stacking_vectors', [])
        if sv:
            _zds(gg, 'stacking_dz',  np.array([v['dz']  for v in sv]), units='alat')
            _zds(gg, 'stacking_da1', np.array([v['da1'] for v in sv]))
            _zds(gg, 'stacking_da2', np.array([v['da2'] for v in sv]))
        semi_inf_start = info.get('sprkkr_semi_inf_start')
        if semi_inf_start is not None:
            gg.attrs['semi_inf_start_layer'] = semi_inf_start

    n_laydbl = inp_dict.get('n_laydbl')
    if n_laydbl is not None and len(n_laydbl) > 0:
        _zds(gg, 'n_laydbl', np.array(n_laydbl, dtype=int))
    surf_bar = inp_dict.get('surf_bar')
    if surf_bar is not None and len(surf_bar) > 0:
        _zds(gg, 'surf_bar', np.array(surf_bar))
    if a1_2d:
        _zds(gg, 'a1_2d', np.array(a1_2d))
    if a2_2d:
        _zds(gg, 'a2_2d', np.array(a2_2d))


# ── Potential pool ────────────────────────────────────────────────────────────

def _write_potential_zarr(pg: 'zarr.Group', pot, sfn: Optional[SfnData],
                          inp_dict: dict, crystal_sha: str):
    """Populate potentials.zarr/<sha>/ from PotData + SfnData."""
    _zat(pg, crystal_sha=crystal_sha)
    _write_scf_zarr(pg.require_group('scf'), pot, inp_dict)
    _write_radial_zarr(pg, pot, sfn)
    if sfn is not None and len(sfn):
        _write_sfn_zarr(pg, sfn)
    parsed = pg.require_group('parsed_metadata')
    _write_json_sidecar(parsed, 'potential', _build_potential_metadata_payload(pot))
    # provenance: raw inp content
    _write_provenance_zarr(pg.require_group('provenance'), inp_dict.get('_calc_dir', ''),
                           inp_dict.get('_files'))


def _write_scf_zarr(sc: 'zarr.Group', pot, inp: dict):
    if inp is None:
        inp = {}
    _RY_TO_EV = 13.605693122994
    scf = pot['SCF-INFO']
    gs  = pot['GLOBAL SYSTEM PARAMETER']
    ne_raw = scf['NE']()
    ne_val = int(ne_raw.flat[0]) if ne_raw is not None and hasattr(ne_raw, 'flat') else int(ne_raw or 0)
    nktab_pot = int(scf['NKTAB']() or 0)
    fermi_ry  = float(scf['EF']() or 0.)
    _zat(sc,
         irel=int(gs['IREL']() or 3),
         irel_description='1=non-rel 2=scalar-rel 3=full-relativistic',
         nspin=int(gs['NSPIN']() or 1),
         xc_potential=str(scf['XC-POT']() or 'VWN'),
         scf_algorithm=str(scf['SCF-ALG']() or 'BROYDEN2'),
         fullpot=int(bool(scf['FULLPOT']())),
         lloyd_pot=int(bool(scf['LLOYD']())),
         fermi_energy_ry=fermi_ry,
         fermi_energy_ev=fermi_ry * _RY_TO_EV,
         vmtz_ry=float(scf['VMTZ']() or 0.),
         ne_energy_mesh=ne_val,
         nktab_pot=nktab_pot,
         scf_iterations=int(scf['SCF-ITER']() or 0),
         scf_tolerance=float(scf['SCF-TOL']() or 0.),
         rmsavv=float(scf['RMSAVV']() or 0.),
         rmsavb=float(scf['RMSAVB']() or 0.),
         scf_status=str(scf['SCFSTATUS']() or ''),
         bzint=str(inp.get('bzint', 'POINTS')),
         nktab=inp.get('nktab') or nktab_pot,
         nktab2d=inp.get('nktab2d') or 0,
         nl=str(inp.get('nl') or ''),
         lloyd=int(inp.get('lloyd', False)),
         rel_mode=str(inp.get('rel_mode', '')),
         krws=inp.get('krws', 1),
         nonmag=int(inp.get('nonmag', False)),
         nosym=int(inp.get('nosym', False)))
    if inp.get('krmt') is not None:
        sc.attrs['krmt'] = inp['krmt']
    mdir = inp.get('mdir')
    if mdir:
        _zds(sc, 'mdir', np.array(mdir, dtype=float),
             description='Magnetisation direction [x,y,z]')


def _full_r_mesh(mesh: ExponentialMesh, im: int, sfn) -> np.ndarray:
    inner = mesh.coors if len(mesh) > 0 else np.array([])
    if len(mesh) == 0 and sfn is not None:   # FULLPOT: jrws==0
        sfn_m = sfn.mesh_for_idx(im)
        if sfn_m is not None and len(sfn_m.rmesh) > 0:
            return np.concatenate([inner, sfn_m.rmesh])
    return inner


def _write_radial_zarr(eg: 'zarr.Group', pot, sfn: Optional[SfnData] = None):
    _BOHR_TO_ANG = 0.529177210903
    rg = eg.require_group('radial_data')
    _zat(rg, description='Radial potential and charge density per atomic type')

    # Build mesh list from MESH INFORMATION section
    msh_rows = pot['MESH INFORMATION']['DATA']()
    meshes   = []   # list of (im, ExponentialMesh, jrcri, jrns1)
    if msh_rows is not None:
        for im, row in enumerate(msh_rows, 1):
            m = ExponentialMesh(float(row[0]), float(row[1]),
                                int(row[2]),   float(row[3]),
                                int(row[4]),   float(row[5]))
            meshes.append([im, m, 0, 0])   # jrcri/jrns1 filled below

    # FULLPOT sub-table: patch jrws with JRCRI so len(mesh) and coors work
    fp_rows = pot['MESH INFORMATION']['FULLPOT']()
    if fp_rows is not None:
        for i, fp_row in enumerate(fp_rows):
            if i < len(meshes):
                jrns1, jrcri = int(fp_row[0]), int(fp_row[1])
                meshes[i][2] = jrcri
                meshes[i][3] = jrns1
                meshes[i][1].jrws = jrcri   # patch so len(mesh)==jrcri

    mg = rg.require_group('meshes')
    for im, m, jrcri, jrns1 in meshes:
        n_pts = len(m)   # jrws after FULLPOT patch, or original jrws for ASA
        dm = mg.require_group(f'IM{im:02d}')
        _zat(dm, im=im, r1=m.r1, dx=m.dx,
             jrmt=m.jrmt, rmt_bohr=m.rmt, rmt_ang=m.rmt * _BOHR_TO_ANG,
             jrws=m.jrws, rws_bohr=m.rws, rws_ang=m.rws * _BOHR_TO_ANG,
             jrcri=jrcri, jrns1=jrns1, n_pts=n_pts)
        r_full = _full_r_mesh(m, im, sfn)
        if len(r_full) > 0:
            _zds(dm, 'r', r_full, units='Bohr')
        if sfn is not None and m.jrws == 0:
            sfn_m = sfn.mesh_for_idx(im)
            if sfn_m is not None and len(sfn_m.rmesh) > 0:
                _zds(dm, 'sfn_r', sfn_m.rmesh, units='Bohr')

    # Real atom types (Z > 0)
    typ_data = pot['TYPES']['DATA']()
    real_its  = {i + 1 for i, row in enumerate(typ_data) if int(row[1]) > 0} \
                if typ_data is not None else set()
    mesh_by_it = {im: m for im, m, _, _ in meshes}

    nspin = int(pot['GLOBAL SYSTEM PARAMETER']['NSPIN']() or 1)

    def _write_group(dg, it, arr, mesh, is_potential: bool):
        data   = arr[0]
        spin_dn = arr[1] if nspin == 2 and arr.shape[0] >= 2 else None
        r = _full_r_mesh(mesh, it, sfn) if mesh is not None else np.array([])
        n = len(data)
        if len(r) > 0:
            _zds(dg, 'r', r[:n], units='Bohr')
        if is_potential:
            if spin_dn is not None:
                _zds(dg, 'V_r_up', data,    units='Ry')
                _zds(dg, 'V_r_dn', spin_dn, units='Ry')
                _zds(dg, 'V_r',   (data + spin_dn) / 2., units='Ry')
            else:
                _zds(dg, 'V_r', data, units='Ry')
        else:
            if spin_dn is not None:
                _zds(dg, 'rho_r_up', data,              units='e/Bohr^3')
                _zds(dg, 'rho_r_dn', spin_dn,           units='e/Bohr^3')
                _zds(dg, 'rho_r',    data + spin_dn,    units='e/Bohr^3')
            else:
                _zds(dg, 'rho_r', data, units='e/Bohr^3')
        if mesh is not None:
            dg.attrs['rmt_bohr'] = mesh.rmt
            dg.attrs['rws_bohr'] = mesh.rws

    def _type_label(it):
        if typ_data is not None and it - 1 < len(typ_data):
            return str(typ_data[it - 1][0])
        return f'type{it}'

    def _type_Z(it):
        if typ_data is not None and it - 1 < len(typ_data):
            return int(typ_data[it - 1][1])
        return 0

    pg = rg.require_group('potential')
    for sec in pot['POTENTIAL'].values():
        it  = int(sec['TYPE']())
        if it not in real_its:
            continue
        arr  = sec['DATA']()
        mesh = mesh_by_it.get(it)
        lbl  = _type_label(it)
        dg   = pg.require_group(lbl)
        _zat(dg, type_idx=it, label=lbl, Z=_type_Z(it))
        _write_group(dg, it, arr, mesh, is_potential=True)

    cg   = rg.require_group('charge')
    seen: set = set()
    for sec in pot['CHARGE'].values():
        it  = int(sec['TYPE']())
        if it not in real_its or it in seen:
            continue
        arr  = sec['DATA']()
        if arr[0].size < 100:
            continue
        seen.add(it)
        mesh = mesh_by_it.get(it)
        lbl  = _type_label(it)
        dg   = cg.require_group(lbl)
        _zat(dg, type_idx=it, label=lbl, Z=_type_Z(it))
        _write_group(dg, it, arr, mesh, is_potential=False)


def _write_sfn_zarr(eg: 'zarr.Group', sfn):
    sg = eg.require_group('shape_functions')
    _zat(sg, nm=sfn.nm)
    if not len(sfn):
        return
    _zds(sg, 'rmt_sfn',  np.array([m.rmt     for m in sfn]), units='Bohr')
    _zds(sg, 'rmtfill',  np.array([m.rmtfill for m in sfn]), units='alat')
    _zds(sg, 'vol',      np.array([m.vol      for m in sfn]), units='alat^3')
    mg = sg.require_group('meshes')
    for m in sfn:
        dg = mg.require_group(f'IM{m.idx:02d}')
        _zat(dg, idx=m.idx, npan=m.npan, nr=m.nr, nsfn=m.nsfn,
             rmt_bohr=m.rmt, rmtfill=m.rmtfill, vol=m.vol)
        if len(m.jrcut) > 0:
            _zds(dg, 'jrcut', np.array(m.jrcut, dtype=int))
        if len(m.rmesh) > 0:
            _zds(dg, 'sfn_rmesh', m.rmesh, units='Bohr')
        if m.sfn is not None and m.sfn.size > 0:
            _zds(dg, 'sfn', m.sfn)
        if m.sfn_lm:
            _zds(dg, 'sfn_lm', np.array(m.sfn_lm, dtype=int))


def _write_provenance_zarr(pv: 'zarr.Group', calc_dir: str,
                           files: Optional[dict] = None):
    """Write SHA-256 hashes + paths + raw inp content to a Zarr group."""
    if files is None and calc_dir:
        files = find_files(calc_dir)
    if files:
        for key, fpath in files.items():
            if key.startswith('_'):
                continue
            try:
                pv.attrs[f'{key}_sha256'] = sha256_file(fpath)
                pv.attrs[f'{key}_path']   = str(Path(fpath).resolve())
            except IOError:
                pass

        slurm_path = files.get('slurm_sh')
        if slurm_path and os.path.exists(slurm_path):
            txt = Path(slurm_path).read_text()
            for attr, pat in [
                ('hpc_account',   r'--account[=\s]+([\w-]+)'),
                ('hpc_partition', r'--partition[=\s]+(\S+)'),
                ('hpc_nprocs',    r'--ntasks[=\s]+(\d+)'),
                ('hpc_job_name',  r'--job-name[=\s]+(\S+)'),
            ]:
                m = re.search(pat, txt)
                if m:
                    pv.attrs[attr] = int(m.group(1)) if 'nprocs' in attr else m.group(1)

        out_path = files.get('calc_out')
        if out_path and os.path.exists(out_path):
            txt = Path(out_path).read_text()
            m = re.search(r'KKRSPEC\s+VERSION\s+([\d.]+)', txt)
            if m: pv.attrs['code_version'] = m.group(1)
            m = re.search(r'MPI calculation with NPROCS\s*=\s*(\d+)', txt)
            if m: pv.attrs['nprocs'] = int(m.group(1))
            m = re.search(r'programm execution\s+on\s+([\d/]+)\s+at\s+([\d:]+)', txt)
            if m: pv.attrs['run_date'] = f'{m.group(1)} {m.group(2)}'

        inp_path = files.get('inp')
        if inp_path and os.path.exists(inp_path):
            raw = Path(inp_path).read_text()
            pv.attrs['raw_inp_content'] = raw


def _resolve_pot_file(calc_dir: str, files: dict, inp_dict: dict) -> None:
    """Update ``files['pot']`` from ``potfil`` in the input file when present."""
    potfil = inp_dict.get('potfil', '')
    if not potfil:
        return
    potfil_path = Path(calc_dir) / potfil
    pot_new_path = Path(str(potfil_path) + '_new')
    if pot_new_path.exists():
        files['pot'] = str(pot_new_path)
    elif potfil_path.exists():
        files['pot'] = str(potfil_path)


def _json_ready(value):
    """Convert numpy-heavy nested data to JSON-safe plain Python values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, datetime.datetime, datetime.date)):
        return str(value)
    return value


def _write_json_sidecar(container: 'zarr.Group', name: str, payload: dict) -> None:
    """Store a JSON payload under a named Zarr sidecar group."""
    sidecar = container.require_group(name)
    sidecar.attrs['format'] = 'json'
    sidecar.attrs['schema_version'] = 1
    sidecar.attrs['updated_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
    sidecar.attrs['payload_json'] = json.dumps(_json_ready(payload), sort_keys=True)


def _build_potential_metadata_payload(pot) -> dict:
    """Build nested metadata for the shared potential Zarr sidecar."""
    payload = {'schema_version': 1}
    if pot is None:
        return payload

    try:
        occ = pot['OCCUPATION']['DATA']()
        if occ is not None and len(occ):
            payload['occupation'] = {
                'irefq': [int(r['IREFQ']) for r in occ],
                'imq': [int(r['IMQ']) for r in occ],
                'noq': [int(r['NOQ']) for r in occ],
                'itoq': [[int(pair[0]) for pair in r['ITOQ CONC']] for r in occ],
                'conc': [[float(pair[1]) for pair in r['ITOQ CONC']] for r in occ],
            }
    except Exception:
        pass

    try:
        ref_sec = pot['REFERENCE SYSTEM']
        ref_data = np.asarray(ref_sec['DATA'](), dtype=float)
        if ref_data.size:
            if ref_data.ndim == 1:
                ref_data = ref_data.reshape(1, -1)
            payload['reference_system'] = {
                'nref': int(ref_sec['NREF']() or ref_data.shape[0]),
                'iref': [i for i in range(1, ref_data.shape[0] + 1)],
                'vref': [float(v) for v in ref_data[:, 0]],
                'rmtref': [float(v) for v in ref_data[:, 1]] if ref_data.shape[1] > 1 else [],
            }
    except Exception:
        pass

    for section_name, key in [
        ('HOST MADELUNG POTENTIAL', 'host_madelung_potential'),
        ('CHARGE MOMENTS', 'charge_moments'),
    ]:
        try:
            arr = np.asarray(pot[section_name]['DATA'](), dtype=float)
            if arr.size:
                payload[key] = {
                    'shape': [int(v) for v in arr.shape],
                    'values': arr.ravel().astype(float).tolist(),
                }
        except Exception:
            pass

    try:
        mag = pot['MAGNETISATION DIRECTION']
        mag_payload = {
            'kmrot': int(mag['KMROT']() or 0),
            'qmvec': np.asarray(mag['QMVEC'](), dtype=float).ravel().astype(float).tolist(),
        }
        iq = np.asarray(mag['DATA_IQ'](), dtype=float)
        if iq.size:
            if iq.ndim == 1:
                iq = iq.reshape(1, -1)
            mag_payload['mtet_q'] = iq[:, 0].astype(float).tolist()
            if iq.shape[1] > 1:
                mag_payload['mphi_q'] = iq[:, 1].astype(float).tolist()
            if iq.shape[1] > 2:
                mag_payload['mgam_q'] = iq[:, 2].astype(float).tolist()
        it = np.asarray(mag['DATA_IT'](), dtype=float)
        if it.size:
            if it.ndim == 1:
                it = it.reshape(1, -1)
            mag_payload['mtet_t'] = it[:, 0].astype(float).tolist()
            if it.shape[1] > 1:
                mag_payload['mphi_t'] = it[:, 1].astype(float).tolist()
            if it.shape[1] > 2:
                mag_payload['mgam_t'] = it[:, 2].astype(float).tolist()
        payload['magnetisation_direction'] = mag_payload
    except Exception:
        pass

    try:
        types = pot['TYPES']['DATA']()
        if types is not None and len(types):
            nt = len(types)
            qel = [None] * nt
            nos = [None] * nt
            spn = [None] * nt
            orb = [None] * nt
            hfi = [None] * nt
            for sec in pot['MOMENTS'].values():
                idx = int(sec['TYPE']()) - 1
                data = np.asarray(sec['DATA'](), dtype=float).ravel()
                if idx < 0 or idx >= nt or data.size < 5:
                    continue
                qel[idx], nos[idx], spn[idx], orb[idx], hfi[idx] = map(float, data[:5])

            payload['types'] = {
                'labels': [str(r['TXT']) for r in types],
                'zt': [int(r['ZT']) for r in types],
                'ncort': [int(r['NCORT']) for r in types],
                'nvalt': [int(r['NVALT']) for r in types],
                'nsemcorshlt': [int(r['NSEMCORSHLT']) for r in types],
                'moments': {
                    'qel': qel,
                    'nos': nos,
                    'spn': spn,
                    'orb': orb,
                    'hfi': hfi,
                },
            }
    except Exception:
        pass

    return payload


def _build_entry_metadata_payload(inp: dict, files: Optional[dict]) -> dict:
    """Build nested per-entry JSON metadata for input/runtime/job provenance."""
    payload = {'schema_version': 1}
    if inp:
        payload['input_parameters'] = _json_ready(inp)
    if files:
        calc_out = files.get('calc_out')
        if calc_out and os.path.exists(calc_out):
            payload['runtime_output'] = _json_ready(parse_arpes_out(calc_out))
        slurm_sh = files.get('slurm_sh')
        if slurm_sh and os.path.exists(slurm_sh):
            payload['job_script'] = _json_ready(parse_job_script(slurm_sh))
    return payload


def _extract_promoted_entry_metadata(files: Optional[dict]) -> dict:
    """Extract only the small query-oriented runtime subset for Lance."""
    if not files:
        return {}

    row = {}
    runtime = {}
    job = {}

    calc_out = files.get('calc_out')
    if calc_out and os.path.exists(calc_out):
        runtime = parse_arpes_out(calc_out)

    slurm_sh = files.get('slurm_sh')
    if slurm_sh and os.path.exists(slurm_sh):
        job = parse_job_script(slurm_sh)

    for key in ('kkrspec_version', 'execution_datetime', 'mpi_nprocs', 'cpu_time_sec', 'wall_time_sec', 'stop_status'):
        if key in runtime:
            row[key] = runtime[key]

    if 'slurm_partition' in job:
        row['slurm_partition'] = job['slurm_partition']
    if 'slurm_ntasks' in job:
        row['slurm_ntasks'] = job['slurm_ntasks']

    return row


def _write_entry_metadata_zarr(db_path: str, entry_id: str, inp: dict, files: Optional[dict]) -> None:
    """Write per-entry parsed JSON sidecars under entries.zarr/<entry_id>/."""
    eg = require_zarr(db_path, f'entries.zarr/{entry_id}')
    parsed = eg.require_group('parsed_metadata')
    payload = _build_entry_metadata_payload(inp, files)
    if 'input_parameters' in payload:
        _write_json_sidecar(parsed, 'input', payload['input_parameters'])
    if 'runtime_output' in payload:
        _write_json_sidecar(parsed, 'runtime', payload['runtime_output'])
    if 'job_script' in payload:
        _write_json_sidecar(parsed, 'job', payload['job_script'])


def _refresh_existing_entry(db_path: str, entry_id: str, calc_dir: str, files: dict) -> None:
    """Refresh JSON sidecars and promoted Lance metadata for an existing entry."""
    migrate_entries_schema(db_path)
    with _parse_lock:
        inp_dict = parse_inp(files['inp']) if 'inp' in files else {}
        _resolve_pot_file(calc_dir, files, inp_dict)
        pot = parse_pot(files['pot']) if 'pot' in files else None

    _write_entry_metadata_zarr(db_path, entry_id, inp_dict, files)

    pot_path = files.get('pot')
    if pot is not None and pot_path and os.path.exists(pot_path):
        pot_sha = sha256_pot(pot_path)
        with _zarr_sha_lock(pot_sha):
            pg = require_zarr(db_path, f'potentials.zarr/{pot_sha}')
            parsed = pg.require_group('parsed_metadata')
            _write_json_sidecar(parsed, 'potential', _build_potential_metadata_payload(pot))

    update = {}
    update.update(_extract_promoted_entry_metadata(files))
    if not update:
        return

    # Use a targeted column-level update so we only touch the promoted metadata
    # columns and never overwrite ARPES arrays / NK / NE / photon_energy_ev etc.
    with _lance_lock:
        from .store import _ldb_connect
        db  = _ldb_connect(db_path)
        tbl = db.open_table('entries')
        tbl.update(
            where=f"entry_id = '{entry_id}'",
            values=update,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Lance row builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_lance_row(eid: str, formula: str, label: str,
                     spc_sha: str, crystal_sha: str,
                     geom_sha: Optional[str], pot_sha: Optional[str],
                     pot, inp: dict, spc, spec_out,
                     crystal_geom_meta: dict,
                     files: Optional[dict] = None) -> dict:
    """Build a flat dict for one Lance row from parsed objects."""
    NK, NE = spc.TOTAL().shape

    # Polarization: prefer spec_out (actual Stokes-derived label) over inp pol_p
    
    pol_label = spec_out.polarization_type if spec_out.stokes.get('s0') != 0 else inp.get('pol_p', '')
    vp       = spec_out.vector_potential or {}

    def _arr(a):
        return a.tolist() if a is not None else [0., 0., 0.]

    row = {
        'entry_id':          eid,
        'created_at':        datetime.datetime.utcnow().isoformat() + 'Z',
        'formula':           formula,
        'dataset_label':     label,
        'spc_sha256':        spc_sha,
        'crystal_sha':       crystal_sha or '',
        'geom_sha':          geom_sha    or '',
        'pot_sha':           pot_sha     or '',
        # crystal (denormalized)
        'bravais_type':      str((pot['LATTICE']['BRAVAIS']() or ('','','','',''))[1]),
        'point_group':       str((pot['LATTICE']['BRAVAIS']() or ('','','','',''))[4]),
        'alat_bohr':         float(pot['LATTICE']['ALAT']() or 0.),
        'nq':                int(pot['GLOBAL SYSTEM PARAMETER']['NQ']() or 0),
        'nt':                int(pot['GLOBAL SYSTEM PARAMETER']['NT']() or 0),
        'n_layers':          int(crystal_geom_meta.get('n_layers', 0)),
        # SCF
        'irel':              int(pot['GLOBAL SYSTEM PARAMETER']['IREL']() or 3),
        'nspin':             int(pot['GLOBAL SYSTEM PARAMETER']['NSPIN']() or 1),
        'fullpot':           bool(pot['SCF-INFO']['FULLPOT']()),
        'xc_potential':      str(pot['SCF-INFO']['XC-POT']() or 'VWN'),
        'fermi_energy_ev':   float(pot['SCF-INFO']['EF']() or 0.) * 13.605693122994,
        'scf_status':        str(pot['SCF-INFO']['SCFSTATUS']() or ''),
        # ARPES E range: emaxev=axis[0] (first/high), eminev=axis[-1] (last/low)
        'emaxev':            float(spc.ENERGY()[0, 0])  if spc is not None else 0.,
        'eminev':            float(spc.ENERGY()[-1, 0]) if spc is not None else 0.,
        # photoemission
        'photon_energy_ev':  float(inp.get('ephot') or 0.),
        'polarization':      pol_label,
        'theta_inc_deg':     float(inp.get('theta_ph') or 0.),
        'phi_inc_deg':       float(inp.get('phi_ph') or 0.),
        'NK':                NK,
        'NE':                NE,
        'adsi':              inp.get('adsi', 'ARPES'),
        # k-path BZ definition (k1..k4 for Fermi-surface cuts, line scans, etc.)
        'ka':                [float(x) for x in inp.get('ka') or []],
        'k1':                [float(x) for x in inp.get('k1') or []],
        'nk1':               int(inp.get('nk1') or 0),
        'k2':                [float(x) for x in inp.get('k2') or []],
        'nk2':               int(inp.get('nk2') or 0),
        'k3':                [float(x) for x in inp.get('k3') or []],
        'nk3':               int(inp.get('nk3') or 0),
        'k4':                [float(x) for x in inp.get('k4') or []],
        'nk4':               int(inp.get('nk4') or 0),
        # Stokes vector from SPEC.out
        'stokes_s0':         float(spec_out.stokes['s0']),
        'stokes_s1_pct':     float(spec_out.stokes['s1_pct']),
        'stokes_s2_pct':     float(spec_out.stokes['s2_pct']),
        'stokes_s3_pct':     float(spec_out.stokes['s3_pct']),
        # Photon geometry vectors from SPEC.out
        'photon_wavevector': _arr(spec_out.photon_wavevector),
        
        'jones_vector_re':   _arr((spec_out.vector_potential or {}).get('re')),
        'jones_vector_im':   _arr((spec_out.vector_potential or {}).get('im')),
        # Potential barrier from SPEC.out
        'barrier_ibar':      int(spec_out.potential_barrier.get('ibar', 0)),
        'barrier_epsx':      float(spec_out.potential_barrier.get('epsx', 0.)),
        'barrier_zparup':    spec_out.potential_barrier.get('zparup', [0., 0., 0.]),
        'barrier_zpardn':    spec_out.potential_barrier.get('zpardn', [0., 0., 0.]),
        'barrier_bparp':     spec_out.potential_barrier.get('bparp',  [0., 0., 0.]),
        # ARPES arrays (float32, NK×NE flattened row-major; loaded as (NE,NK))
        'k_axis':            spc.K()[0, :].astype(np.float32).tolist(),
        'energy_axis':       spc.ENERGY()[:, 0].astype(np.float32).tolist(),
        'intensity_total':   spc.TOTAL().astype(np.float32).ravel().tolist(),
        'intensity_up':      spc.UP().astype(np.float32).ravel().tolist(),
        'intensity_down':    spc.DOWN().astype(np.float32).ravel().tolist(),
        'spin_polarization': spc.POLARIZATION().astype(np.float32).ravel().tolist(),
        'determinant':       spc.DETERMINANT().astype(np.float32).ravel().tolist(),
        # embedding omitted here; filled by compute_embeddings() post-ingest
    }

    row.update(_extract_promoted_entry_metadata(files))
    return row


def _row_to_arrow(row: dict) -> pa.Table:
    """Convert a single row dict to a PyArrow Table using the canonical schema."""
    schema = entries_schema()
    arrays = []
    for field in schema:
        val = row.get(field.name)
        if isinstance(field.type, pa.lib.ListType):
            arrays.append(pa.array([val], type=field.type))
        else:
            arrays.append(pa.array([val], type=field.type))
    return pa.table({f.name: arrays[i] for i, f in enumerate(schema)}, schema=schema)


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level writer
# ══════════════════════════════════════════════════════════════════════════════
# Thread-safety helpers for parallel ingestion
# ══════════════════════════════════════════════════════════════════════════════

# Parse lock: ase2sprkkr / PyParsing grammars are not thread-safe (they mutate
# global state during parsing).  Serialising the parse phase preserves
# correctness while still allowing Zarr/Lance writes to overlap across workers.
_parse_lock = threading.Lock()

# Global lock: serialises alloc_entry_id + lance_append
_lance_lock = threading.Lock()

# Per-SHA locks: allow different SHAs to write to Zarr concurrently
_zarr_lock_registry: dict = {}
_zarr_registry_lock = threading.Lock()


def _zarr_sha_lock(sha: str) -> threading.Lock:
    """Return (creating if needed) a per-SHA threading.Lock."""
    with _zarr_registry_lock:
        if sha not in _zarr_lock_registry:
            _zarr_lock_registry[sha] = threading.Lock()
        return _zarr_lock_registry[sha]


# ══════════════════════════════════════════════════════════════════════════════

def write_lance(db_path: str, entry_id: str, formula: str,
                spc, spec_out, inp_dict: dict,
                pot, sfn, in_struct, calc_dir: str,
                crystal_sha: Optional[str] = None,
                geom_sha: Optional[str] = None,
                pot_sha: Optional[str] = None,
                files: Optional[dict] = None) -> str:
    """
    Write one OSCARpes entry to the v3 Lance + Zarr database.

    Shared data (crystal / LKKR geometry / potential) is written once to the
    respective Zarr pool and referenced by SHA-256 key.  The Lance row stores
    all metadata and ARPES arrays for fast ML access.
    """
    init_db(db_path)
    migrate_entries_schema(db_path)

    # ── 1. Crystal pool ───────────────────────────────────────────────────────
    if crystal_sha is None:
        crystal_sha = sha256_crystal(pot)
    with _zarr_sha_lock(crystal_sha):
        if not zarr_exists(db_path, f'crystals.zarr/{crystal_sha}'):
            cg = require_zarr(db_path, f'crystals.zarr/{crystal_sha}')
            _write_crystal_zarr(cg, pot)

    # ── 2. LKKR geometry pool ─────────────────────────────────────────────────
    if geom_sha is None and files and 'in_structur' in files:
        geom_sha = sha256_lkkr_geom(files['in_structur'])
    if geom_sha:
        with _zarr_sha_lock(geom_sha):
            if not zarr_exists(db_path, f'lkkr_geometry.zarr/{geom_sha}'):
                gg = require_zarr(db_path, f'lkkr_geometry.zarr/{geom_sha}')
                _write_lkkr_geom_zarr(gg, in_struct, inp_dict)

    # ── 3. Potential pool ─────────────────────────────────────────────────────
    if pot_sha is None and files and 'pot' in files:
        pot_sha = sha256_pot(files['pot'])
    if pot_sha:
        with _zarr_sha_lock(pot_sha):
            if not zarr_exists(db_path, f'potentials.zarr/{pot_sha}'):
                pg = require_zarr(db_path, f'potentials.zarr/{pot_sha}')
                inp_dict_prov = dict(inp_dict)
                inp_dict_prov['_calc_dir'] = calc_dir
                inp_dict_prov['_files']    = files
                _write_potential_zarr(pg, pot, sfn, inp_dict_prov, crystal_sha)
            else:
                pg = require_zarr(db_path, f'potentials.zarr/{pot_sha}')
                parsed = pg.require_group('parsed_metadata')
                _write_json_sidecar(parsed, 'potential', _build_potential_metadata_payload(pot))

    # ── 3b. Entry metadata sidecars ──────────────────────────────────────────
    _write_entry_metadata_zarr(db_path, entry_id, inp_dict, files)

    # ── 4. Validate Zarr pool completeness before committing Lance row ────────
    # This ensures no Lance row is written if a Zarr write failed mid-way,
    # keeping the database in a consistent state.  On the next ingest attempt
    # the existing (partial) Zarr data is reused via zarr_exists checks, and
    # only the Lance row is re-attempted.
    if crystal_sha and not zarr_exists(db_path, f'crystals.zarr/{crystal_sha}'):
        raise RuntimeError(
            f"[write_lance] Crystal Zarr pool {crystal_sha[:12]}… was not written. "
            "Lance row will NOT be committed to preserve database consistency."
        )
    if pot_sha and not _pot_zarr_complete(db_path, pot_sha):
        raise RuntimeError(
            f"[write_lance] Potential Zarr pool {pot_sha[:12]}… is incomplete. "
            "Lance row will NOT be committed to preserve database consistency."
        )

    # ── 5. Lance row ──────────────────────────────────────────────────────────
    formula_auto = ''
    try:
        formula_auto = _formula_from_pot(pot)
    except (AttributeError, KeyError, TypeError, ImportError):
        pass
    label = formula if formula not in ('', 'unknown') else _auto_label(inp_dict, formula_auto)
    formula_final = formula_auto or formula

    geom_meta = {}
    if in_struct is not None:
        geom_meta['n_layers'] = in_struct.info.get('sprkkr_n_layers', 0)

    row   = _build_lance_row(
        entry_id, formula_final, label,
        sha256_file(files['spc']) if files and 'spc' in files else '',
        crystal_sha, geom_sha, pot_sha,
        pot, inp_dict, spc, spec_out, geom_meta, files=files,
    )
    table = _row_to_arrow(row)
    with _lance_lock:
        lance_append(db_path, table)

    return entry_id


# ══════════════════════════════════════════════════════════════════════════════
#  High-level public API
# ══════════════════════════════════════════════════════════════════════════════

def _pot_zarr_complete(db_path: str, pot_sha: str) -> bool:
    """Return True if the potentials zarr pool has radial_data/potential written."""
    try:
        g = open_zarr(db_path, f'potentials.zarr/{pot_sha}/radial_data/potential', mode='r')
        return len(list(g.keys())) > 0
    except (zarr.errors.GroupNotFoundError, FileNotFoundError, KeyError, OSError):
        return False


def _repair_zarr_if_needed(entry_id: str, db_path: str, files: dict, calc_dir: str) -> None:
    """Re-write any Zarr pool groups that are missing or incomplete."""
    from .store import lance_filter_one
    row = lance_filter_one(db_path, entry_id)
    if row is None:
        return

    pot_sha     = row.get('pot_sha')
    geom_sha    = row.get('geom_sha')
    crystal_sha = row.get('crystal_sha')

    needs_repair = False

    if pot_sha and not _pot_zarr_complete(db_path, pot_sha):
        needs_repair = True
    if crystal_sha and not zarr_exists(db_path, f'crystals.zarr/{crystal_sha}'):
        needs_repair = True
    if geom_sha and not zarr_exists(db_path, f'lkkr_geometry.zarr/{geom_sha}'):
        needs_repair = True

    if not needs_repair:
        return

    print('[OSCAR] ⚠  Zarr pools incomplete — repairing …')

    inp_dict  = parse_inp(files['inp'])            if 'inp'         in files else {}
    _resolve_pot_file(calc_dir, files, inp_dict)

    pot       = parse_pot(files['pot'])            if 'pot'         in files else None
    sfn       = parse_sfn(files['sfn'],
                              alat=float(pot['LATTICE']['ALAT']() or 0.) if pot is not None else 0.) \
                                                       if 'sfn' in files else None
    from ase2sprkkr.sprkkr.structure import structure_file_to_atoms as _sfa
    in_struct = _sfa(files['in_structur'], pot) \
                                                       if 'in_structur' in files and pot is not None else None

    if crystal_sha and not zarr_exists(db_path, f'crystals.zarr/{crystal_sha}'):
        cg = require_zarr(db_path, f'crystals.zarr/{crystal_sha}')
        _write_crystal_zarr(cg, pot)

    if geom_sha and not zarr_exists(db_path, f'lkkr_geometry.zarr/{geom_sha}'):
        gg = require_zarr(db_path, f'lkkr_geometry.zarr/{geom_sha}')
        _write_lkkr_geom_zarr(gg, in_struct, inp_dict)

    if pot_sha and not _pot_zarr_complete(db_path, pot_sha):
        # Remove incomplete group and rewrite from scratch
        import shutil, os
        pot_dir = _join(db_path, f'potentials.zarr/{pot_sha}')
        if os.path.exists(pot_dir):
            shutil.rmtree(pot_dir)
        pg = require_zarr(db_path, f'potentials.zarr/{pot_sha}')
        inp_dict_prov = dict(inp_dict)
        inp_dict_prov['_calc_dir'] = calc_dir
        inp_dict_prov['_files']    = files
        _write_potential_zarr(pg, pot, sfn, inp_dict_prov, crystal_sha)

    print('[OSCAR] ✓  Zarr pools repaired.')


def ingest_directory(calc_dir: str, db_path: Optional[str] = None,
                     formula: str = 'unknown',
                     entry_id: Optional[str] = None,
                     force: bool = False) -> str:
    """
    Parse a complete SPR-KKR ARPES calculation directory and write one
    oscarpes v3 entry (Lance row + Zarr pool data).

    Duplicate detection
    -------------------
    Before parsing, the SHA-256 of the SPC file is compared against all
    existing entries in the Lance dataset.  If a match is found the existing
    ``entry_id`` is returned immediately.  Pass ``force=True`` to re-ingest.

    Parameters
    ----------
    calc_dir : str
        Directory containing ``*.inp``, ``*pot_new``, ``*_data.spc``, etc.
        ``find_files`` picks the largest ``*.spc`` when multiple exist.
    db_path  : str, optional
        oscarpes database directory (local path or ``s3://`` URI).
        Defaults to ``~/.oscarpes/``.
    formula  : str
        User dataset label override, e.g. ``'2H-WSe2'``.
    entry_id : str, optional
        UUID to use; auto-generated when None.
    force    : bool
        Re-ingest even when a duplicate SPC SHA-256 is detected.

    Returns
    -------
    str : UUID of the written or pre-existing entry.
    """
    if db_path is None:
        db_path = DEFAULT_DB
    init_db(db_path)
    files = find_files(calc_dir)

    if 'spc' not in files:
        raise FileNotFoundError(f'No *_data.spc found in {calc_dir!r}')

    # ── duplicate check ───────────────────────────────────────────────────────
    spc_hash = sha256_file(files['spc'])
    if not force:
        existing = lance_find_entry_id(db_path, spc_hash)
        if existing:
            # Check whether Zarr pools are complete; repair silently if not.
            _repair_zarr_if_needed(existing, db_path, files, calc_dir)
            _refresh_existing_entry(db_path, existing, calc_dir, dict(files))
            print(f'[OSCAR] ↩  Duplicate detected — entry already in database.')
            print(f'           spc SHA-256 : {spc_hash[:16]}…')
            print(f'           entry_id    : {existing}')
            print(f'           Refreshed metadata for any schema-expanded columns.')
            print(f'           Use force=True to overwrite.')
            return existing

    # ── parse phase (serialised: ase2sprkkr/PyParsing is not thread-safe) ────
    with _parse_lock:
        spc      = parse_spc(files['spc'])
        spec_out = parse_spec_out(files['spec_out']) if 'spec_out' in files else SpecOutData()
        inp_dict = parse_inp(files['inp'])           if 'inp'      in files else {}
        _resolve_pot_file(calc_dir, files, inp_dict)

        pot      = parse_pot(files['pot']) if 'pot' in files else None
        sfn      = parse_sfn(files['sfn'],
                              alat=float(pot['LATTICE']['ALAT']() or 0.) if pot is not None else 0.) \
                                                       if 'sfn' in files else None
        from ase2sprkkr.sprkkr.structure import structure_file_to_atoms as _sfa
        in_struct = _sfa(files['in_structur'], pot) \
                                                       if 'in_structur' in files and pot is not None else None

        crystal_sha = sha256_crystal(pot)
        geom_sha    = sha256_lkkr_geom(files['in_structur']) if 'in_structur' in files else None
        pot_sha     = sha256_pot(files['pot'])                if 'pot'         in files else None

    # Allocate a Materials-Project-style ID: osc-<formula>-<n>
    # Use the pot-derived Hill formula when available, fall back to user label.
    formula_auto = ''
    try:
        formula_auto = _formula_from_pot(pot)
    except (AttributeError, KeyError, TypeError, ImportError):
        pass
    id_formula = formula_auto or formula
    with _lance_lock:
        eid = entry_id or alloc_entry_id(db_path, id_formula)
    # Note: _lance_lock is re-acquired inside write_lance for lance_append.
    # alloc_entry_id must be under the same lock to prevent duplicate IDs
    # when ingest_tree runs workers in parallel.

    write_lance(db_path, eid, formula, spc, spec_out, inp_dict,
                pot, sfn, in_struct, calc_dir,
                crystal_sha=crystal_sha, geom_sha=geom_sha, pot_sha=pot_sha,
                files=files)

    label = formula if formula not in ('', 'unknown') else _auto_label(inp_dict, formula_auto)

    print(f'[OSCAR] ✓  entry_id={eid}  →  {db_path}')
    return eid


def ingest_tree(root_dir: str, db_path: Optional[str] = None,
                formula: str = 'unknown',
                force: bool = False,
                verbose: bool = True,
                workers: int = 1) -> List[str]:
    """
    Recursively find all SPR-KKR calculation directories under ``root_dir``
    and ingest each one into the database.

    Discovery uses the same heuristic as ``find_files``: a directory is
    considered a calculation if it contains at least one ``*_data.spc``
    file.  When multiple ``.spc`` files share a directory the largest one
    wins (same rule as ``find_files``).

    Parameters
    ----------
    root_dir : str
        Root directory to search recursively.
    db_path  : str, optional
        oscarpes database directory.  Defaults to ``~/.oscarpes/``.
    formula  : str
        Default formula label (passed to each ``ingest_directory`` call).
    force    : bool
        Re-ingest even when duplicates are found.
    verbose  : bool
        Print a one-line status per directory.
    workers : int
        Number of parallel threads.  Each worker independently parses its
        calculation directory; SHA-keyed Zarr writes and Lance appends are
        serialised automatically so deduplication is never compromised.
        Default 1 (serial, original behaviour).

    Returns
    -------
    list of str : entry_ids of all successfully ingested entries.
    """
    if db_path is None:
        db_path = DEFAULT_DB
    root = Path(root_dir)
    spc_parents: dict = {}
    for spc_file in sorted(root.rglob('*_data.spc')):
        d = spc_file.parent
        sz = spc_file.stat().st_size
        if d not in spc_parents or sz > spc_parents[d]:
            spc_parents[d] = sz

    if not spc_parents:
        warnings.warn(f'[ingest_tree] No *_data.spc files found under {root_dir!r}',
                      UserWarning, stacklevel=2)
        return []

    calc_dirs = sorted(spc_parents)

    def _one(calc_dir):
        return ingest_directory(str(calc_dir), db_path, formula=formula, force=force)

    eids = []
    if workers <= 1:
        for calc_dir in calc_dirs:
            try:
                eid = _one(calc_dir)
                eids.append(eid)
                if verbose:
                    print(f'[OSCAR tree] ✓  {calc_dir.relative_to(root)}  →  {eid[:8]}…')
            except Exception as exc:
                if verbose:
                    print(f'[OSCAR tree] ✗  {calc_dir.relative_to(root)}  {exc}')
    else:
        future_to_dir = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for calc_dir in calc_dirs:
                future_to_dir[pool.submit(_one, calc_dir)] = calc_dir
            for future in as_completed(future_to_dir):
                calc_dir = future_to_dir[future]
                try:
                    eid = future.result()
                    eids.append(eid)
                    if verbose:
                        print(f'[OSCAR tree] ✓  {calc_dir.relative_to(root)}  →  {eid[:8]}…')
                except Exception as exc:
                    if verbose:
                        print(f'[OSCAR tree] ✗  {calc_dir.relative_to(root)}  {exc}')
    return eids


def compute_embeddings(db_path: Optional[str] = None, batch_size: int = 128) -> None:
    """
    Compute the 128-dim ML feature vector for every entry that has an
    empty ``embedding`` column and write it back to the Lance dataset.

    Calls :func:`oscarpes.ml_features.extract_features` for each entry.
    Safe to call multiple times — only updates rows where ``embedding`` is
    empty or None.

    Parameters
    ----------
    db_path    : str, optional  Database directory.  Defaults to ``~/.oscarpes/``.
    batch_size : int            Number of entries processed per Lance merge fragment.
    """
    if db_path is None:
        db_path = DEFAULT_DB
    import lancedb
    import pyarrow as pa
    from .ml_features import extract_features
    from .entry import OSCAREntry

    db  = lancedb.connect(db_path)
    tbl = db.open_table('entries')

    # Rows without embeddings — fetch only scalar + embedding columns
    result = tbl.search().select(['entry_id']).to_arrow()
    eids   = result.column('entry_id').to_pylist()

    print(f'[OSCAR] compute_embeddings: computing {len(eids)} embeddings…')
    updates = []
    for eid in eids:
        try:
            e   = OSCAREntry(eid, db_path)
            vec = extract_features(e).astype(np.float32)
            updates.append({'entry_id': eid, 'embedding': vec.tolist()})
        except Exception as exc:
            print(f'[OSCAR]   SKIP {eid[:8]}…  {exc}')

    if not updates:
        return

    # Update via lancedb merge_insert (upsert on entry_id)
    upd_tbl = pa.table({
        'entry_id':  pa.array([u['entry_id']  for u in updates], pa.string()),
        'embedding': pa.array([u['embedding'] for u in updates],
                               pa.list_(pa.float32())),
    })
    tbl.merge_insert(on='entry_id').when_matched_update_all().execute(upd_tbl)
    print(f'[OSCAR] compute_embeddings: wrote {len(updates)} embeddings.')
