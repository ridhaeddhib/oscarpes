"""
oscarpes.nomad_plugin.schema
================================
NOMAD metainfo schema for SPR-KKR ARPES calculations.

Hierarchy
---------
::

    run
    ├── ModelSystem  [nomad-simulations — AtomicCell + AtomsState]
    ├── ModelMethod
    │   └── KKRMethod  [CUSTOM extends DFT]
    │       └── RadialPotential[]  per atom type
    └── TheoreticalARPES  [CUSTOM output]
        ├── PhotonSource
        ├── DetectorGeometry
        ├── SemiInfiniteSurface   ← miller_hkl, iq_at_surf, work_function_ev here
        └── ARPESOutput
"""
from __future__ import annotations

import numpy as np

try:
    from nomad.metainfo import MSection, Quantity, SubSection, MEnum, Section
    from nomad_simulations.schema_packages.model_method import DFT
    _NOMAD_AVAILABLE = True
except ImportError:
    # Allow import without nomad-lab installed (schema unusable but no crash)
    _NOMAD_AVAILABLE = False
    MSection = object
    Quantity = SubSection = MEnum = Section = lambda *a, **kw: None
    DFT = object


# ══════════════════════════════════════════════════════════════════════════════
#  KKR Method
# ══════════════════════════════════════════════════════════════════════════════

class RadialPotential(MSection):
    """Radial potential and charge density for one atomic type.

    Stored in ``potentials.zarr/<sha>/radial_data/`` — exposed here as a
    NOMAD SubSection so it can be searched and visualised in the NOMAD UI.
    """
    m_def = Section(label='RadialPotential')

    label    = Quantity(type=str,
                        description='Atomic type label (e.g. Au, W, Se)')
    Z        = Quantity(type=int,
                        description='Atomic number')
    type_idx = Quantity(type=int,
                        description='IT index in the SPR-KKR potential file')
    rmt_bohr = Quantity(type=float, unit='bohr',
                        description='Muffin-tin radius')
    rws_bohr = Quantity(type=float, unit='bohr',
                        description='Wigner-Seitz radius')
    r_mesh   = Quantity(type=np.dtype('f8'), shape=['*'], unit='bohr',
                        description='Radial mesh r [Bohr]')
    V_r      = Quantity(type=np.dtype('f8'), shape=['*'], unit='Ry',
                        description='Self-consistent crystal potential V(r) [Ry]')
    rho_r    = Quantity(type=np.dtype('f8'), shape=['*'],
                        unit='1/bohr^3',
                        description='Charge density ρ(r) [e/Bohr³]')


class KKRMethod(DFT):
    """KKR-specific quantities extending the nomad-simulations DFT section.

    Adds relativistic treatment, Lloyd formula flag, SCF convergence info,
    and radial potential data per atom type.
    """
    m_def = Section(extends_base_section=True)

    irel = Quantity(
        type=MEnum(1, 2, 3),
        description='Relativistic treatment: 1=non-rel, 2=scalar-rel, 3=full-rel',
    )
    xc_potential = Quantity(
        type=str,
        description='Exchange-correlation potential (e.g. VWN, PBE, LDA)',
    )
    fullpot = Quantity(
        type=bool,
        description='Full potential (True) vs muffin-tin (False)',
    )
    lloyd = Quantity(
        type=bool,
        description='Lloyd formula used for total energy / charge',
    )
    bzint = Quantity(
        type=str,
        description='BZ integration method (e.g. POINTS, TETRA)',
    )
    fermi_energy_ev = Quantity(
        type=float, unit='eV',
        description='Self-consistent Fermi energy',
    )
    fermi_energy_ry = Quantity(
        type=float,
        description='Self-consistent Fermi energy in Rydberg (natural SPR-KKR unit)',
    )
    scf_status = Quantity(
        type=str,
        description='CONVERGED / NOT_CONVERGED',
    )
    scf_iterations = Quantity(
        type=int,
        description='Number of SCF iterations performed',
    )
    rmsavv = Quantity(
        type=float,
        description='Root-mean-square change of valence potential (convergence criterion)',
    )
    radial_potentials = SubSection(sub_section=RadialPotential, repeats=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TheoreticalARPES subsections
# ══════════════════════════════════════════════════════════════════════════════

class PhotonSource(MSection):
    """Theoretical photon source parameters.

    Aligns with the NeXus NXsource vocabulary for cross-comparison with
    experimental ARPES datasets (NXmpes_arpes).
    """
    m_def = Section(label='PhotonSource')

    photon_energy = Quantity(
        type=float, unit='eV',
        description='Photon energy hν',
    )
    polarization = Quantity(
        type=MEnum('LH', 'LV', 'C+', 'C-', 'general'),
        description='Polarization state (LH=linear horiz, LV=linear vert, '
                    'C+=left-circular, C-=right-circular)',
    )
    theta_inc_deg = Quantity(
        type=float, unit='degree',
        description='Polar angle of incidence θ',
    )
    phi_inc_deg = Quantity(
        type=float, unit='degree',
        description='Azimuthal angle of incidence φ',
    )
    stokes_s0 = Quantity(
        type=float,
        description='Stokes parameter S0 (total intensity)',
    )
    stokes_s1_pct = Quantity(
        type=float, unit='%',
        description='Stokes S1 (linear horizontal/vertical)',
    )
    stokes_s2_pct = Quantity(
        type=float, unit='%',
        description='Stokes S2 (linear ±45°)',
    )
    stokes_s3_pct = Quantity(
        type=float, unit='%',
        description='Stokes S3 circular degree: −100%=C+ (left), +100%=C−',
    )


class DetectorGeometry(MSection):
    """Detector / final-state geometry parameters.

    Covers the electron-optics side of the ARPES calculation
    (emission angle ranges, final-state model, inner potential).
    """
    m_def = Section(label='DetectorGeometry')

    theta_el_range = Quantity(
        type=np.dtype('f8'), shape=[2], unit='degree',
        description='Polar emission angle range [θ_min, θ_max]',
    )
    phi_el_range = Quantity(
        type=np.dtype('f8'), shape=[2], unit='degree',
        description='Azimuthal emission angle range [φ_min, φ_max]',
    )
    final_state_model = Quantity(
        type=str,
        description='Final-state model (e.g. free-electron, damped-plane-wave)',
    )
    imv_initial_ev = Quantity(
        type=float, unit='eV',
        description='Imaginary part of the inner potential (initial state)',
    )
    imv_final_ev = Quantity(
        type=float, unit='eV',
        description='Imaginary part of the inner potential (final state)',
    )


class SemiInfiniteSurface(MSection):
    """Semi-infinite surface geometry — the physical sample surface.

    Contains both crystallographic orientation (miller_hkl) and
    the actual semiinfinite geometry (layer positions, atom coordinates).
    IQ_AT_SURF and MILLER_HKL from the SPR-KKR input file live here
    because they define *which surface was computed*, not how emission
    is collected.

    Work function is also a surface property (depends on material + hkl).
    """
    m_def = Section(label='SemiInfiniteSurface')

    miller_hkl = Quantity(
        type=np.dtype('i4'), shape=[3],
        description='Surface orientation — Miller indices (h, k, l)',
    )
    iq_at_surf = Quantity(
        type=int,
        description='IQ index of the surface-layer atomic site (SPR-KKR convention)',
    )
    work_function_ev = Quantity(
        type=float, unit='eV',
        description='Surface work function (material + orientation dependent)',
    )
    n_layers = Quantity(
        type=int,
        description='Number of inequivalent layers in the semi-infinite stack',
    )
    alat_2d = Quantity(
        type=float, unit='bohr',
        description='2D in-plane lattice parameter a_lat',
    )
    layer_z_positions = Quantity(
        type=np.dtype('f8'), shape=['*'], unit='bohr',
        description='z-position of first atom per layer [Bohr]',
    )
    semiinfinite_positions_bohr = Quantity(
        type=np.dtype('f8'), shape=['*', 3], unit='bohr',
        description='Cartesian positions of all semiinfinite atoms [Bohr]',
    )
    semiinfinite_positions_ang = Quantity(
        type=np.dtype('f8'), shape=['*', 3], unit='angstrom',
        description='Cartesian positions of all semiinfinite atoms [Å]',
    )


class ARPESOutput(MSection):
    """Theoretical ARPES output arrays.

    Aligns with the NXmpes_arpes NeXus application definition for
    cross-comparison with experimental ARPES datasets:
    - k_axis        ↔ NXmpes_arpes/data/@axes[k_parallel]
    - energy_axis   ↔ NXmpes_arpes/data/@axes[energy]
    - intensity_*   ↔ NXmpes_arpes/data/
    """
    m_def = Section(label='ARPESOutput')

    NK = Quantity(type=int, description='Number of k∥ grid points')
    NE = Quantity(type=int, description='Number of energy grid points')

    k_axis = Quantity(
        type=np.dtype('f4'), shape=['NK'], unit='1/angstrom',
        description='k∥ grid [Å⁻¹]',
    )
    energy_axis = Quantity(
        type=np.dtype('f4'), shape=['NE'], unit='eV',
        description='Binding energy axis [eV], E_Fermi = 0',
    )
    intensity_total = Quantity(
        type=np.dtype('f4'), shape=['NK', 'NE'],
        description='Total photoemission intensity I(k∥, E)',
    )
    intensity_up = Quantity(
        type=np.dtype('f4'), shape=['NK', 'NE'],
        description='Spin-up photoemission intensity I↑(k∥, E)',
    )
    intensity_down = Quantity(
        type=np.dtype('f4'), shape=['NK', 'NE'],
        description='Spin-down photoemission intensity I↓(k∥, E)',
    )
    spin_polarization = Quantity(
        type=np.dtype('f4'), shape=['NK', 'NE'], unit='%',
        description='Spin polarization P = (I↑−I↓)/(I↑+I↓) [%]',
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level ARPES output section
# ══════════════════════════════════════════════════════════════════════════════

class TheoreticalARPES(MSection):
    """Top-level output section for one SPR-KKR ARPES calculation.

    Contains four sibling subsections that together describe the full
    photoemission calculation — aligns with NXmpes_arpes vocabulary:

    - PhotonSource       ↔ NXsource
    - DetectorGeometry   ↔ NXdetector / NXmonochromator (final-state side)
    - SemiInfiniteSurface↔ NXsample (surface orientation + geometry)
    - ARPESOutput        ↔ NXdata
    """
    m_def = Section(label='TheoreticalARPES')

    photon_source      = SubSection(sub_section=PhotonSource)
    detector_geometry  = SubSection(sub_section=DetectorGeometry)
    semi_infinite_surf = SubSection(sub_section=SemiInfiniteSurface)
    arpes_output       = SubSection(sub_section=ARPESOutput)
