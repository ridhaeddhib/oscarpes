OSCAREntry
==========

:class:`~oscarpes.entry.OSCAREntry` is the central data object.  It is
returned by :meth:`~oscarpes.database.OSCARDatabase.__getitem__` and by
:meth:`~oscarpes.entry.OSCAREntry.from_directory`.

On construction it loads:

1. The **Lance row** (metadata scalars + ARPES arrays, all in memory).
2. The **Zarr crystal pool** referenced by ``crystal_sha``.
3. The **Zarr LKKR geometry pool** referenced by ``geom_sha``.
4. The **Zarr potential pool** (SCF data) referenced by ``pot_sha``.

Radial data and shape functions are loaded on-demand via ``get_radial_*``
methods.

All raw file parsing is done during ingestion by the **ase2sprkkr** backend;
an ``OSCAREntry`` object never reads SPR-KKR files directly.

Data attributes
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Attribute
     - Type
     - Source / description
   * - ``e.arpes``
     - :class:`~oscarpes.entry.ARPESData`
     - ARPES intensity arrays and axes — from ``*_data.spc`` via ``ARPESOutputFile``
   * - ``e.crystal``
     - :class:`~oscarpes.entry.CrystalData`
     - Bulk lattice, Bravais type, atom types — from ``*.pot`` via ``Potential``
   * - ``e.lkkr_geometry``
     - :class:`~oscarpes.entry.LKKRGeometryData`
     - Layer stack, barrier, 2D basis vectors — from ``in_structur.inp`` via ``structure_file_to_atoms``
   * - ``e.scf``
     - :class:`~oscarpes.entry.SCFData`
     - Fermi energy, XC, convergence — from ``*.pot`` and ``*.inp``
   * - ``e.photon``
     - :class:`~oscarpes.entry.PhotonSourceData`
     - Photon energy, polarisation, Stokes vector, wavevector — from ``*.inp`` + ``*_SPEC.out``
   * - ``e.photoemission``
     - :class:`~oscarpes.entry.SampleData`
     - Work function, IMV, k-path definition (ka, k1, nk1), E range
   * - ``e.structure``
     - :class:`~oscarpes.entry.StructureData`
     - Legacy combined view (synthesised from crystal + lkkr_geometry)

ARPESData
---------

``e.arpes`` wraps the ARPES arrays parsed from ``*_data.spc`` by ase2sprkkr's
``ARPESOutputFile``.

.. code-block:: python

   a = e.arpes

   # Shape
   a.NK, a.NE                   # number of k-points, energy points

   # 2-D intensity arrays  (NE, NK)  float32
   a.intensity_total            # total ARPES cross section
   a.intensity_up               # spin-up channel
   a.intensity_down             # spin-down channel
   a.spin_polarization          # P = (up − down) / (up + down)
   a.determinant                # KKR determinant (diagnostic)

   # 1-D axes
   a.energy_axis                # (NE,) eV, relative to Fermi level
   a.k_axis                     # (NK,) Å⁻¹

   # Derived
   a.spin_polarization_masked   # P with zero-intensity points masked to NaN
   a.extent                     # (k0, k1, e0, e1) for imshow

   # Cuts
   E, I          = a.edc(0.0)            # EDC at k‖ = 0 Å⁻¹
   k, I          = a.mdc(-0.5)           # MDC at E = −0.5 eV
   E, Iup, Idn   = a.edc_spin(0.0)       # spin-resolved EDC
   fm            = a.fermi_map(e_tol=0.05)  # intensity integrated near E_F

CrystalData
-----------

Loaded from ``crystals.zarr/<crystal_sha>/``.  Data originates from the
``*.pot`` file parsed by ase2sprkkr's ``Potential.from_file()``.

.. code-block:: python

   c = e.crystal
   c.bravais_type        # 'hexagonal', 'cubic', etc.
   c.bravais_index       # SPR-KKR integer Bravais index
   c.point_group         # 'D_6h', 'O_h', etc.
   c.alat_bohr           # lattice constant a (Bohr)
   c.alat_angstrom       # lattice constant a (Å)
   c.nq                  # number of inequivalent sites
   c.nt                  # number of atom types (including vacancies)
   c.type_labels         # list of atom type labels, e.g. ['W', 'Se', 'Se']
   c.a1, c.a2, c.a3     # lattice vectors (dimensionless × alat)

LKKRGeometryData
----------------

Loaded from ``lkkr_geometry.zarr/<geom_sha>/``.  Data originates from
``in_structur.inp`` parsed by ase2sprkkr's ``structure_file_to_atoms()``.

.. code-block:: python

   g = e.lkkr_geometry
   g.n_layers                    # number of layers in the semi-infinite stack
   g.alat_2d                     # 2D in-plane lattice constant (Å)
   g.layer_z_positions           # z-coordinates of each layer (alat)
   g.semiinfinite_positions_bohr # (N, 3) Cartesian atom positions (Bohr)
   g.a1_2d, g.a2_2d             # 2D in-plane lattice vectors (dimensionless)
   g.barrier_ibar                # potential barrier type (0, 1, or 2)
   g.barrier_epsx                # barrier epsilon (eV)
   g.barrier_zparup              # barrier z-parameters for spin-up
   g.barrier_zpardn              # barrier z-parameters for spin-down

SCFData
-------

Loaded from ``potentials.zarr/<pot_sha>/scf/``.  Data originates from
``*.pot`` (via ``Potential``) and ``*.inp`` (via ``InputParameters``).

.. code-block:: python

   s = e.scf
   s.fermi_energy_ev     # Fermi energy (eV)
   s.fermi_energy_ry     # Fermi energy (Ry)
   s.xc_potential        # exchange-correlation functional, e.g. 'VWN'
   s.irel                # 1 = non-relativistic, 2 = scalar, 3 = full-relativistic
   s.fullpot             # True = full-potential, False = ASA
   s.lloyd_pot           # True = Lloyd formula used in SCF
   s.nspin               # 1 = non-magnetic, 2 = magnetic
   s.scf_iterations      # number of SCF iterations
   s.scf_tolerance       # SCF convergence tolerance
   s.rmsavv              # RMS charge change (last iteration)
   s.rmsavb              # RMS Madelung potential change
   s.scf_status          # 'CONVERGED', 'NOT_CONVERGED', etc.
   s.vmtz_ry             # muffin-tin zero (Ry)
   s.nktab_pot           # k-mesh size used in the SCF calculation
   s.krws                # 0/1: use Wigner–Seitz radius for sphere construction

PhotonSourceData
----------------

Loaded from Lance scalars.  Source files: ``*.inp`` (via ``InputParameters``)
and ``*_SPEC.out`` (via ase2sprkkr's ``SpecResult``).

.. code-block:: python

   ph = e.photon           # also accessible as e.photon_source
   ph.photon_energy_ev     # photon energy (eV)
   ph.theta_inc_deg        # photon polar incidence angle (degrees)
   ph.phi_inc_deg          # photon azimuthal incidence angle (degrees)
   ph.polarization_label   # 'LH', 'LV', 'C+', 'C-', 'LD', ...
   ph.stokes_s0            # total intensity
   ph.stokes_s1_pct        # linear horizontal polarisation (%)
   ph.stokes_s2_pct        # linear diagonal polarisation (%)
   ph.stokes_s3_pct        # circular polarisation (%, −100=C+, +100=C−)
   ph.photon_wavevector    # (3,) photon wave vector [Å⁻¹]
   ph.vector_potential_re  # (3,) Jones vector Re part
   ph.vector_potential_im  # (3,) Jones vector Im part

SampleData (photoemission)
--------------------------

Loaded from Lance scalars.  Source: ``*.inp`` via ``InputParameters``.

.. code-block:: python

   pe = e.photoemission    # also accessible as e.sample
   pe.ne                   # number of energy points
   pe.nk                   # number of k-points (from nk1)
   pe.ka                   # BZ k-path origin vector (Å⁻¹ or alat⁻¹)
   pe.k1                   # BZ k-path translation vector
   pe.work_function_ev     # work function (eV)
   pe.imv_initial_ev       # imaginary inner potential Γ_i (eV)
   pe.imv_final_ev         # imaginary part of final state (eV)
   pe.final_state_model    # 'FEGFINAL' or 'FP'
   pe.iq_at_surf           # index of surface atomic sphere
   pe.miller_hkl           # Miller indices of the surface

Radial data (on-demand)
-----------------------

Radial data is stored in ``potentials.zarr/<pot_sha>/radial_data/``.  It is
not loaded on construction; use the following methods:

.. code-block:: python

   # Radial potential V(r)
   r, V = e.get_radial_potential('W')   # label must match type_labels
   r, V = e.get_radial_potential('Se')

   # Radial charge density ρ(r)
   r, rho = e.get_radial_charge('W')

   # Mesh information (RMT, RWS, number of radial points, …)
   meshes = e.get_mesh_info()   # list of dicts, one per atom type
   for m in meshes:
       print(m['label'], m['rmt_bohr'], m['rws_bohr'])

   # Real atom types (exclude vacancies / empty spheres)
   labels = e.real_atom_labels()   # ['W', 'Se']

   # Raw .inp file content stored in provenance
   raw_inp = e.get_raw_inp()

ASE Atoms reconstruction
------------------------

The entry can reconstruct an ASE ``Atoms`` object from stored crystal / LKKR
geometry data.  ase2sprkkr's ``SPRKKRAtoms.promote_ase_atoms()`` is used to
attach SPR-KKR metadata:

.. code-block:: python

   # Semi-infinite slab (uses semiinfinite_positions_bohr)
   atoms = e.to_ase_atoms()
   print(atoms.get_chemical_formula())

   # Bulk unit cell (from crystals.zarr sites)
   bulk  = e.to_ase_atoms(semiinfinite=False)
   print(bulk.cell)

Saving / copying entries
------------------------

Copy an entry (Lance row + all referenced Zarr pool groups) to another database:

.. code-block:: python

   e.save('/data/subset_db/')

Loading from a directory (convenience constructor)
---------------------------------------------------

.. code-block:: python

   e = OSCAREntry.from_directory(
       '/data/WSe2_calc',
       db_path='/data/oscar_db/',
       formula='2H-WSe2',
   )
   # Equivalent to ingest_directory + OSCAREntry(eid, db_path)

   # Force re-ingest even if SPC SHA already exists
   e = OSCAREntry.from_directory('/data/WSe2_calc', '/data/oscar_db/', force=True)

Provenance
----------

File paths, SHA-256 hashes, SLURM parameters, and the raw ``*.inp`` content
are stored in ``potentials.zarr/<pot_sha>/provenance/``:

.. code-block:: python

   paths = e.get_source_paths()
   # {'inp': '/data/WSe2_calc/kkrspec.inp',
   #  'pot': '/data/WSe2_calc/WSe2.pot_new', ...}
