"""
oscarpes.nomad_plugin.parser
================================
NOMAD MatchingParser for SPR-KKR ARPES calculations.

Wraps ``oscarpes.parsers`` (no additional parsing logic needed) and
populates the custom metainfo schema defined in ``schema.py``.

Archive layout produced
-----------------------
::

    archive.run[0]
    ├── program          (KKRSPEC / SPR-KKR)
    ├── model_system[0]  (AtomicCell from pot lattice)
    └── model_method[0]  (KKRMethod — SCF, radial potentials)

    archive.data         (TheoreticalARPES — photon, detector, surface, output)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    from nomad.parsing import MatchingParser
    _NOMAD_AVAILABLE = True
except ImportError:
    _NOMAD_AVAILABLE = False

    class MatchingParser:  # type: ignore[misc]
        """Stub when nomad-lab is not installed."""


_RY_TO_EV   = 13.605693122994
_BOHR_TO_ANG = 0.529177210903


class SPRKKRParser(MatchingParser):  # type: ignore[misc]
    """
    First KKR photoemission NOMAD parser.

    Matches ``*.inp`` ARPES task files produced by SPR-KKR / xband and
    populates the archive from the sibling ``.pot``, ``*_data.spc``,
    ``*_SPEC.out``, and ``in_structur.inp`` files found in the same
    directory.

    No file-format logic is implemented here — all parsing is delegated
    to ``oscarpes.parsers``.
    """

    mainfile_name_re = r'.*\.inp$'
    mainfile_mime_re = r'text/.*'

    # ── entry point ───────────────────────────────────────────────────────────

    def parse(self, mainfile: str, archive, logger) -> None:
        from oscarpes.parsers import find_files, parse_inp, parse_pot, parse_in_structur
        from .schema import (
            KKRMethod, RadialPotential,
            TheoreticalARPES, PhotonSource, DetectorGeometry,
            SemiInfiniteSurface, ARPESOutput,
        )

        calc_dir = Path(mainfile).parent
        files    = find_files(str(calc_dir))

        # ── parse raw files ───────────────────────────────────────────────────
        inp = parse_inp(mainfile)

        pot = None
        if 'pot' in files:
            try:
                pot = parse_pot(files['pot'])
            except Exception as exc:
                logger.warning(f'SPRKKRParser: pot parsing failed: {exc}')

        in_str = None
        if 'in_structur' in files:
            try:
                in_str = parse_in_structur(files['in_structur'])
            except Exception as exc:
                logger.warning(f'SPRKKRParser: in_structur parsing failed: {exc}')

        spc = None
        if 'spc' in files:
            try:
                from oscarpes.parsers import parse_spc
                spc = parse_spc(files['spc'])
            except Exception as exc:
                logger.warning(f'SPRKKRParser: spc parsing failed: {exc}')

        spec_out = None
        if 'spec_out' in files:
            try:
                from oscarpes.parsers import parse_spec_out
                spec_out = parse_spec_out(files['spec_out'])
            except Exception as exc:
                logger.warning(f'SPRKKRParser: spec_out parsing failed: {exc}')

        # ── archive.run  (ModelSystem + KKRMethod) ────────────────────────────
        self._populate_run(archive, inp, pot, logger)

        # ── archive.data  (TheoreticalARPES) ─────────────────────────────────
        arpes_sec = TheoreticalARPES()
        arpes_sec.photon_source      = self._make_photon_source(inp, spec_out, logger)
        arpes_sec.detector_geometry  = self._make_detector_geometry(inp, logger)
        arpes_sec.semi_infinite_surf = self._make_surface(inp, in_str, logger)

        if spc is not None:
            out = self._make_arpes_output(spc, logger)
            if out is not None:
                arpes_sec.arpes_output = out

        archive.data = arpes_sec

    # ── run / model_system / KKRMethod ────────────────────────────────────────

    def _populate_run(self, archive, inp: dict, pot, logger) -> None:
        try:
            from nomad_simulations.schema_packages.general import Run, Program
        except ImportError:
            return

        run = Run()
        run.program = Program(name='KKRSPEC / SPR-KKR')
        archive.run.append(run)

        if pot is not None:
            self._add_model_system(run, pot, logger)
            self._add_kkr_method(run, inp, pot, logger)

    def _add_model_system(self, run, pot, logger) -> None:
        try:
            from nomad_simulations.schema_packages.model_system import (
                ModelSystem, AtomicCell, AtomsState,
            )
            ms   = ModelSystem()
            cell = AtomicCell()
            # Lattice vectors converted to metres (NOMAD SI units)
            _bohr_to_m = 5.29177210903e-11
            for vec in (pot.a1, pot.a2, pot.a3):
                cell.positions = np.array(vec) * pot.alat_bohr * _bohr_to_m
            ms.cell.append(cell)

            for site in pot.sites:
                sym = (site.get('label') or 'X')[:2].capitalize()
                ms.atoms_state.append(AtomsState(chemical_symbol=sym))

            run.model_system.append(ms)
        except Exception as exc:
            logger.warning(f'SPRKKRParser: ModelSystem failed: {exc}')

    def _add_kkr_method(self, run, inp: dict, pot, logger) -> None:
        from .schema import KKRMethod, RadialPotential

        try:
            method = KKRMethod()
            method.irel            = pot.irel
            method.xc_potential    = pot.xc_pot
            method.fullpot         = pot.fullpot
            method.lloyd           = pot.lloyd
            method.fermi_energy_ev = pot.fermi_ev
            method.fermi_energy_ry = pot.fermi_ry
            method.scf_status      = pot.scf_status
            method.scf_iterations  = pot.scf_iter
            method.rmsavv          = pot.rmsavv

            if inp.get('bzint'):
                method.bzint = str(inp['bzint'])

            # Radial potentials — one per atom type
            for i, rd in enumerate(pot.potentials):
                rp = RadialPotential()
                rp.label    = getattr(rd, 'label', '')
                rp.Z        = int(getattr(rd, 'Z', 0))
                rp.type_idx = i + 1

                r = getattr(rd, 'r_mesh', None)
                v = getattr(rd, 'V_r',   None)
                q = getattr(rd, 'rho_r', None)
                if r is not None: rp.r_mesh = np.asarray(r, dtype=np.float64)
                if v is not None: rp.V_r    = np.asarray(v, dtype=np.float64)
                if q is not None: rp.rho_r  = np.asarray(q, dtype=np.float64)

                mesh = pot.mesh_for_type(i + 1)
                if mesh is not None:
                    rp.rmt_bohr = float(getattr(mesh, 'rmt', 0.))
                    rp.rws_bohr = float(getattr(mesh, 'rws', 0.))

                method.radial_potentials.append(rp)

            run.model_method.append(method)
        except Exception as exc:
            logger.warning(f'SPRKKRParser: KKRMethod failed: {exc}')

    # ── TheoreticalARPES subsections ──────────────────────────────────────────

    def _make_photon_source(self, inp: dict, spec_out, logger):
        from .schema import PhotonSource
        ph = PhotonSource()

        try:
            if inp.get('ephot')    is not None: ph.photon_energy  = float(inp['ephot'])
            if inp.get('theta_ph') is not None: ph.theta_inc_deg  = float(inp['theta_ph'])
            if inp.get('phi_ph')   is not None: ph.phi_inc_deg    = float(inp['phi_ph'])

            if spec_out is not None:
                ph.stokes_s0     = float(spec_out.stokes_s0)
                ph.stokes_s1_pct = float(spec_out.stokes_s1_pct)
                ph.stokes_s2_pct = float(spec_out.stokes_s2_pct)
                ph.stokes_s3_pct = float(spec_out.stokes_s3_pct)
                ph.polarization  = spec_out.polarization_type   # 'LH','LV','C+','C-'
            elif inp.get('icirc') is not None:
                # Fallback: derive polarization from ICIRC flag in .inp
                icirc = int(inp['icirc'])
                ph.polarization = {0: 'LH', 1: 'C+', -1: 'C-'}.get(icirc, 'general')
        except Exception as exc:
            logger.warning(f'SPRKKRParser: PhotonSource failed: {exc}')

        return ph

    def _make_detector_geometry(self, inp: dict, logger):
        from .schema import DetectorGeometry
        det = DetectorGeometry()

        try:
            if inp.get('imv_ini_ev') is not None:
                det.imv_initial_ev = float(inp['imv_ini_ev'])
            if inp.get('imv_fin_ev') is not None:
                det.imv_final_ev = float(inp['imv_fin_ev'])
            if inp.get('final_state_model'):
                det.final_state_model = str(inp['final_state_model'])

            # Emission angle range from k-scan grid
            k1 = inp.get('k1'); k2 = inp.get('k2')
            if k1 is not None and k2 is not None:
                det.theta_el_range = np.array([float(k1), float(k2)], dtype=np.float64)
        except Exception as exc:
            logger.warning(f'SPRKKRParser: DetectorGeometry failed: {exc}')

        return det

    def _make_surface(self, inp: dict, in_str, logger):
        from .schema import SemiInfiniteSurface
        surf = SemiInfiniteSurface()

        try:
            if inp.get('miller_hkl') is not None:
                surf.miller_hkl = np.array(inp['miller_hkl'], dtype=np.int32)
            if inp.get('iq_at_surf') is not None:
                surf.iq_at_surf = int(inp['iq_at_surf'])
            if inp.get('ework_ev') is not None:
                surf.work_function_ev = float(inp['ework_ev'])

            if in_str is not None:
                surf.n_layers = in_str.n_layers
                if in_str.alat is not None:
                    surf.alat_2d = float(in_str.alat)
                z = in_str.z_positions
                if z.size > 0 and in_str.alat:
                    surf.layer_z_positions = z * in_str.alat   # [alat] → [Bohr]
                pos_bohr = in_str.atom_cart_positions_bohr()
                if pos_bohr.size > 0:
                    surf.semiinfinite_positions_bohr = pos_bohr
                    surf.semiinfinite_positions_ang  = pos_bohr * _BOHR_TO_ANG
        except Exception as exc:
            logger.warning(f'SPRKKRParser: SemiInfiniteSurface failed: {exc}')

        return surf

    def _make_arpes_output(self, spc, logger):
        from .schema import ARPESOutput
        try:
            out = ARPESOutput()

            # K and ENERGY are 1D axes; TOTAL etc. are 2D (NK × NE)
            k_arr = np.asarray(spc.K,      dtype=np.float32).ravel()
            e_arr = np.asarray(spc.ENERGY, dtype=np.float32).ravel()
            out.NK          = int(k_arr.size)
            out.NE          = int(e_arr.size)
            out.k_axis      = k_arr
            out.energy_axis = e_arr

            total = np.asarray(spc.TOTAL, dtype=np.float32)
            if total.ndim == 1:
                total = total.reshape(out.NK, out.NE)
            out.intensity_total = total

            for attr_src, attr_dst in [
                ('UP',          'intensity_up'),
                ('DOWN',        'intensity_down'),
                ('POLARIZATION','spin_polarization'),
            ]:
                arr = getattr(spc, attr_src, None)
                if arr is not None:
                    a = np.asarray(arr, dtype=np.float32)
                    if a.ndim == 1:
                        a = a.reshape(out.NK, out.NE)
                    setattr(out, attr_dst, a)

            return out
        except Exception as exc:
            logger.warning(f'SPRKKRParser: ARPESOutput failed: {exc}')
            return None
