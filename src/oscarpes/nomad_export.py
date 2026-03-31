"""
oscarpes.nomad_export
==============================
FAIRmat / NOMAD-compatible export of OSCARpes entries.

Extends the existing ase2sprkkr NOMAD binding
(ase2sprkkr.bindings.nomad.nomad) with OSCAR-specific workflow YAML
and file packaging for SPR-KKR ARPES calculations.

Usage
-----
::

    from oscarpes.nomad_export import export_entry, export_database

    # Export one entry
    export_entry(entry, 'wse2_arpes.zip')

    # Export entire database
    export_database(db, 'atlas_upload.zip')

Schema mapping
--------------
NOMAD section               ← OSCAR source
─────────────────────────────────────────────
ElectronicStructureMethod   ← scf/{irel, xc_potential, fullpot}
Photoemission               ← photoemission_params/{hν, pol, Stokes}
KohnShamStates (proxy)      ← arpes_data/{intensity_total, k_axis, E_axis}
GeometryOptimization        ← structure/{alat, sites, type_labels}

The generated archive conforms to NOMAD's workflow YAML format:
  https://nomad-lab.eu/prod/v1/docs/howto/customization/workflows.html
"""
from __future__ import annotations
import os
import io
import json
import zipfile
import datetime
import tempfile
from pathlib import Path
from typing import Optional, List, Union
import numpy as np

from .entry    import OSCAREntry
from .database import OSCARDatabase

_BOHR_TO_ANG = 0.529177210903
_RY_TO_EV    = 13.605693122994


# ── NOMAD metainfo section names ──────────────────────────────────────────────

_NOMAD_SECTIONS = {
    'workflow':   'nomad.datamodel.metainfo.workflow.TaskReference',
    'system':     'runschema.system.System',
    'method':     'runschema.method.Method',
    'calculation':'runschema.calculation.Calculation',
}


class _NpEncoder(json.JSONEncoder):
    """Handle numpy scalar types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def _nomad_archive_dict(entry: OSCAREntry) -> dict:
    """
    Build the NOMAD archive dict for one OSCAREntry.

    Returns a Python dict that, when serialised to YAML, is a valid
    NOMAD workflow archive conforming to the FAIRmat ARPES metainfo.
    """
    sc  = entry.scf
    ph  = entry.photon
    a   = entry.arpes
    st  = entry.structure

    # ── model_system ──────────────────────────────────────────────────────────
    model_system = {
        'm_def': 'nomad.datamodel.metainfo.simulation.system.AtomicCell',
        'label': entry.formula,
        'lattice_constant_bohr': float(st.alat_bohr),
        'lattice_constant_angstrom': float(st.alat_angstrom),
        'bravais_type': st.bravais_type,
        'point_group': st.point_group,
        'n_atoms': st.nq,
        'n_types': st.nt,
        'type_labels': st.type_labels or [],
        'n_layers': st.n_layers,
    }
    if st.slab_positions_bohr is not None:
        model_system['atom_positions_bohr'] = st.slab_positions_bohr.tolist()
        model_system['atom_positions_angstrom'] = (
            st.slab_positions_bohr * _BOHR_TO_ANG).tolist()

    # ── model_method ──────────────────────────────────────────────────────────
    model_method = {
        'm_def': 'nomad.datamodel.metainfo.simulation.method.DFT',
        'electronic_structure': 'KKR Green-function (SPR-KKR)',
        'relativistic_treatment': {1:'non-relativistic',
                                   2:'scalar-relativistic',
                                   3:'full-relativistic'}.get(sc.irel, str(sc.irel)),
        'exchange_correlation_functional': sc.xc_potential,
        'full_potential': bool(sc.fullpot),
        'lloyd_formula': bool(sc.lloyd),
        'fermi_energy_eV': sc.fermi_energy_ev,
        'fermi_energy_Ry': sc.fermi_energy_ry,
        'mtz_potential_Ry': sc.vmtz_ry,
        'n_k_points_scf': sc.nktab,
        'n_energy_mesh_scf': sc.ne_energy_mesh,
        'scf_iterations': sc.scf_iterations,
        'scf_convergence_criterion': sc.scf_tolerance,
        'scf_rms_val_electrons': sc.rmsavv,
        'scf_rms_core_electrons': sc.rmsavb,
        'scf_status': sc.scf_status,
        'code': 'KKRSPEC / SPR-KKR',
    }

    # ── photoemission section ─────────────────────────────────────────────────
    photoemission = {
        'm_def': 'nomad.datamodel.metainfo.eln.experiment.Photoemission',
        'photon_energy_eV': ph.photon_energy_ev,
        'polarization': ph.polarization_label,
        'theta_incidence_deg': ph.theta_inc_deg,
        'phi_incidence_deg': ph.phi_inc_deg,
        'work_function_eV': ph.work_function_ev,
        'imv_final_state_eV': ph.imv_final_ev,
        'final_state_model': ph.final_state_model,
        'surface_site_IQ': ph.iq_at_surf,
        'stokes_vector': {
            's0': ph.stokes_s0,
            's1_pct': ph.stokes_s1_pct,
            's2_pct': ph.stokes_s2_pct,
            's3_pct': ph.stokes_s3_pct,
            'note': 's3=-100% = pure left-circular (C+)',
        },
    }
    if ph.miller_hkl is not None:
        photoemission['miller_hkl'] = ph.miller_hkl.tolist()
    if ph.photon_wavevector is not None:
        photoemission['photon_wavevector_Bohr_inv'] = ph.photon_wavevector.tolist()
    if ph.vector_potential_re is not None:
        photoemission['jones_vector_re'] = ph.vector_potential_re.tolist()
        photoemission['jones_vector_im'] = ph.vector_potential_im.tolist()

    # ── outputs ───────────────────────────────────────────────────────────────
    outputs = {
        'm_def': 'nomad.datamodel.metainfo.simulation.calculation.Calculation',
        'energy': {'fermi_eV': sc.fermi_energy_ev},
        'arpes_data': {
            'k_parallel_axis_Ang_inv': a.k_axis.tolist(),
            'energy_axis_eV':          a.energy_axis.tolist(),
            'NK': a.NK,
            'NE': a.NE,
            'intensity_shape': list(a.intensity_total.shape),
            'intensity_max':   float(a.intensity_total.max()),
            'energy_reference': 'E_Fermi = 0 eV',
        },
        'k_grid': {
            'nk1': int(a.NK),
        },
    }

    # ── full archive ──────────────────────────────────────────────────────────
    archive = {
        'definitions': {
            'name': 'OSCARpes',
            'description': ('Ab-initio Theoretical Library for ARPES Spectroscopy.'
                            ' SPR-KKR KKR Green-function photoemission calculation.'),
        },
        'data': {
            'm_def': 'nomad.datamodel.metainfo.simulation.run.Run',
            'program': {
                'name': 'KKRSPEC / SPR-KKR',
                'version': entry.schema_version,
            },
            'model_system':   [model_system],
            'model_method':   [model_method],
            'photoemission':  [photoemission],
            'outputs':        [outputs],
        },
        'metadata': {
            'entry_id':     entry.entry_id,
            'formula':      entry.formula,
            'created_at':   entry.created_at,
            'atlas_schema': entry.schema_version,
            'datasets':     ['OSCARpes'],
            'comment':      (f'{entry.formula} ARPES calculation at '
                             f'{ph.photon_energy_ev} eV, {ph.polarization_label}'),
        },
    }
    return archive


def _workflow_yaml(entries: List[OSCAREntry], name: str = 'OSCARpes') -> str:
    """
    Generate NOMAD workflow YAML linking SCF → ARPES tasks.

    This follows the NOMAD TaskReference workflow schema.
    """
    import yaml

    tasks   = []
    inputs  = []
    outputs = []

    for e in entries:
        entry_name = f'{e.formula}_{e.photon.photon_energy_ev:.0f}eV_{e.photon.polarization_label}'
        task = {
            'name': f'ARPES of {entry_name}',
            'm_def': _NOMAD_SECTIONS['workflow'],
            'inputs': [
                {'name': e.formula, 'section': f'../upload/archive/mainfile/{e.entry_id}.json#/data/model_system[0]'},
                {'name': 'KKR method', 'section': f'../upload/archive/mainfile/{e.entry_id}.json#/data/model_method[0]'},
            ],
            'outputs': [
                {'name': f'ARPES {entry_name}', 'section': f'../upload/archive/mainfile/{e.entry_id}.json#/data/outputs[0]'},
            ],
        }
        tasks.append(task)
        inputs.extend(task['inputs'])
        outputs.extend(task['outputs'])

    workflow = {
        'workflow2': {
            'm_def': _NOMAD_SECTIONS['workflow'],
            'name': name,
            'inputs':  inputs,
            'outputs': outputs,
            'tasks':   tasks,
        }
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True,
                     sort_keys=False)


def export_entry(entry: OSCAREntry,
                 zip_path: str,
                 include_raw_inp: bool = True) -> str:
    """
    Export one OSCAREntry as a NOMAD-compatible zip archive.

    Contents
    --------
    <entry_id>.json           NOMAD archive JSON
    workflow.archive.yaml     NOMAD workflow YAML
    input.inp                 (optional) original SPR-KKR .inp file

    Parameters
    ----------
    entry           : OSCAREntry to export
    zip_path        : output path for the ZIP file
    include_raw_inp : include the raw .inp file from provenance

    Returns
    -------
    zip_path : str
    """
    import json

    archive_dict = _nomad_archive_dict(entry)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Main archive JSON
        zf.writestr(
            f'{entry.entry_id}.json',
            json.dumps(archive_dict, indent=2, ensure_ascii=False, cls=_NpEncoder)
        )

        # Workflow YAML
        zf.writestr('workflow.archive.yaml', _workflow_yaml([entry]))

        # Raw .inp content from provenance (Zarr-backed)
        if include_raw_inp:
            try:
                raw = entry.get_raw_inp()
                if raw:
                    zf.writestr('input.inp', raw)
            except Exception:
                pass

        # README
        readme = _make_readme(entry)
        zf.writestr('README.txt', readme)

    return zip_path


def export_database(db: OSCARDatabase,
                    zip_path: str,
                    formula_contains: Optional[str] = None) -> str:
    """
    Export all (or filtered) entries in an OSCARDatabase as a NOMAD upload.

    Parameters
    ----------
    db               : OSCARDatabase instance
    zip_path         : output ZIP path
    formula_contains : optional formula filter (e.g. 'WSe')

    Returns
    -------
    zip_path : str
    """
    import json

    entries = db.find(formula=formula_contains) if formula_contains else db.find()
    if not entries:
        raise ValueError('No entries found to export')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for e in entries:
            archive_dict = _nomad_archive_dict(e)
            fname = f'{e.entry_id}.json'
            zf.writestr(fname, json.dumps(archive_dict, indent=2, ensure_ascii=False))

        # Shared workflow YAML
        zf.writestr('workflow.archive.yaml', _workflow_yaml(entries))

        # Summary JSON
        summary = {
            'atlas_schema': '1.0.0',
            'n_entries': len(entries),
            'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'entries': [
                {
                    'entry_id':        e.entry_id,
                    'formula':         e.formula,
                    'photon_energy_ev':e.photon.photon_energy_ev,
                    'polarization':    e.photon.polarization_label,
                    'scf_status':      e.scf.scf_status,
                }
                for e in entries
            ]
        }
        zf.writestr('atlas_summary.json',
                    json.dumps(summary, indent=2, ensure_ascii=False, cls=_NpEncoder))

        zf.writestr('README.txt', _make_db_readme(entries))

    return zip_path


def _make_readme(entry: OSCAREntry) -> str:
    ph = entry.photon; sc = entry.scf; st = entry.structure
    return f"""OSCARpes  —  FAIRmat / NOMAD Export
=========================================

Entry ID    : {entry.entry_id}
Formula     : {entry.formula}
Created     : {entry.created_at}
Schema      : OSCARpes v{entry.schema_version}

Calculation
-----------
Code        : KKRSPEC / SPR-KKR
Relativistic: IREL={sc.irel} (3=full-relativistic)
XC potential: {sc.xc_potential}
Full-pot    : {sc.fullpot}
Lloyd       : {sc.lloyd}
SCF status  : {sc.scf_status}  ({sc.scf_iterations} iterations)
Fermi energy: {sc.fermi_energy_ev:.5f} eV

Structure
---------
Lattice     : {st.bravais_type}  {st.point_group}
a_lat       : {st.alat_bohr:.4f} Bohr = {st.alat_angstrom:.4f} Å
Sites NQ/NT : {st.nq} / {st.nt}
Slab layers : {st.n_layers}

Photoemission
--------------
hν          : {ph.photon_energy_ev} eV
Polarization: {ph.polarization_label}
θ_inc       : {ph.theta_inc_deg}°  φ_inc: {ph.phi_inc_deg}°
Stokes s3   : {ph.stokes_s3_pct}%  (−100%=C+, +100%=C−)
Work fn     : {ph.work_function_ev} eV
Im V_fin    : {ph.imv_final_ev} eV
Final state : {ph.final_state_model}

Files in this archive
---------------------
{entry.entry_id}.json   — NOMAD archive JSON
workflow.archive.yaml   — NOMAD workflow YAML
input.inp               — Original SPR-KKR input file
README.txt              — This file

NOMAD upload
------------
Upload all files in this archive to nomad-lab.eu/upload.
The workflow.archive.yaml provides task linkage for the NOMAD UI.

References
----------
OSCARpes: part of ase2sprkkr (https://github.com/ase2sprkkr/ase2sprkkr)
SPR-KKR    : Ebert et al., Rep. Prog. Phys. 74, 096501 (2011)
FAIRmat    : https://www.fairmat-nfdi.eu
"""


def _make_db_readme(entries: List[OSCAREntry]) -> str:
    lines = [
        'OSCARpes  —  FAIRmat / NOMAD Database Export',
        '=' * 50,
        '',
        f'Exported: {datetime.datetime.utcnow().isoformat()}Z',
        f'Entries : {len(entries)}',
        '',
        'Entries',
        '-------',
    ]
    for e in entries:
        lines.append(f'  {e.entry_id[:8]}  {e.formula:<12}  '
                     f'hν={e.photon.photon_energy_ev:.0f}eV  '
                     f'pol={e.photon.polarization_label}  '
                     f'EF={e.scf.fermi_energy_ev:.3f}eV')
    lines += ['', 'Each entry has its own <uuid>.json NOMAD archive.',
              'The workflow.archive.yaml links all entries.',
              '',
              'Upload the entire ZIP to https://nomad-lab.eu/upload']
    return '\n'.join(lines)
