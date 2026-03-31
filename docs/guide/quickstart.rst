Quick start
===========

This guide walks through a complete workflow: ingesting a finished SPR-KKR
ARPES calculation, loading the entry, accessing all data sub-objects, and
producing the standard visualisations.

Prerequisites
-------------

You need ``ase2sprkkr`` installed (the parsing backend) for the ingestion
step.  All other operations work without it:

.. code-block:: bash

   pip install "oscarpes[sprkkr,ase,ml]"

File layout expected by the ingester
-------------------------------------

A SPR-KKR ARPES calculation directory should contain:

.. code-block:: text

   calc_dir/
     ARPES.inp            ← ARPES task parameters 
     sample.pot_new         ← self-consistent potential  (*pot_new or *.pot)
     sample_data.spc        ← ARPES cross sections   (*_data.spc)
     sample_SPEC.out        ← photon / Stokes info   (*_SPEC.out)    
     sample.out             ← KKRSPEC runtime / compiler output       [optional]
     kkrscf.sh              ← batch script / SLURM job file           [optional]
     sample.sfn             ← shape function data    (*.sfn)          [optional]
     in_structur.inp        ← LKKR layer geometry                     

The :func:`~oscarpes.parsers.find_files` function automatically discovers
all these files; you only need to point it at the directory.

Step 1 — Ingest a single calculation
--------------------------------------

.. code-block:: python

   from oscarpes.ingest import ingest_directory

   eid = ingest_directory(
       calc_dir='/data/Se2W_C+_60eV',
       # db_path omitted here: default is ~/.oscarpes/
       formula='Se2W',
   )
   print(eid)   # e.g.  osc-Se2W-1

Under the hood, ``ingest_directory`` uses the **ase2sprkkr** parsing backend
to read every file:

* ``parse_spc`` (``ARPESOutputFile.from_file``) extracts the
  ``I(k‖, E)`` ARPES intensity arrays.
* ``parse_spec_out`` (``SpecResult``) extracts Stokes parameters, photon
  wavevector, Jones vector, and potential barrier parameters.
* ``parse_inp`` (``InputParameters.from_file``) extracts the ARPES task
  parameters (photon energy, angles, polarisation, k-path definition …).
* ``parse_pot`` (``Potential.from_file``) extracts the self-consistent
  potential, Fermi energy, SCF convergence, and radial mesh information.
* ``parse_sfn`` (``read_shape_function``) reads the shape function for
  each muffin-tin / Voronoi cell.
* ``structure_file_to_atoms`` reads the layer geometry into an
  ``SPRKKRAtoms`` object.

Duplicate detection is automatic: if you call ``ingest_directory`` twice on
the same directory the SHA-256 of the SPC file is compared and the existing
``entry_id`` is returned. The duplicate path also refreshes parsed JSON
sidecars for entry/runtime/job and shared potential metadata.

Step 2 — Ingest a whole experiment tree
-----------------------------------------

.. code-block:: python

   from oscarpes.ingest import ingest_tree

   eids = ingest_tree(
       root_dir='/data/Se2W_experiments/',
       db_path='/data/oscar_db/',
       formula='Se2W',
   )
   print(f'Ingested {len(eids)} calculations')

Step 3 — Open the database
---------------------------

.. code-block:: python

   from oscarpes import OSCARDatabase

   db = OSCARDatabase('/data/oscar_db/')
   print(db)

Example output::

   OSCARDatabase('/data/oscar_db/', 3 entries)
     ────────────────────────────────────────────────────────────
     osc-Se2W-1
       Formula   : Se2W
       Bravais   : hexagonal  point group D_6h
       Potential : FullPot  irel=3  non-magnetic
       Photon Source      : hν=60.0 eV  pol=C+
       ARPES Cross Section: E=[0.00, -1.25] eV  k=[-1.500, 1.500] Å⁻¹  (150×150)
     ────────────────────────────────────────────────────────────
     osc-Se2W-2
       ...

Step 4 — Load an entry
-----------------------

.. code-block:: python

   e = db['osc-Se2W-1']
   print(e)

The entry object loads:

* **Lance** — row-oriented metadata and ARPES arrays (fast columnar read).
* **Zarr crystals pool** — lattice parameters, Bravais type, atom sites.
* **Zarr LKKR geometry pool** — semi-infinite layer stack, 2D lattice.
* **Zarr potentials pool** — SCF data, radial potential V(r), charge ρ(r),
  shape functions, and parsed potential JSON sidecars.
* **Zarr entries sidecars** — per-entry input, runtime, and job metadata as JSON.

Step 5 — Access the data
-------------------------

**ARPES spectra** (from ``*_data.spc`` via ase2sprkkr ``ARPESOutputFile``):

.. code-block:: python

   a = e.arpes                      # ARPESData wrapper
   a.intensity_total                # (NE, NK) float32
   a.intensity_up                   # spin-up channel
   a.intensity_down                 # spin-down channel
   a.spin_polarization              # P = (up − down)/(up + down)
   a.energy_axis                    # 1-D  (eV, below EF)
   a.k_axis                         # 1-D  (Å⁻¹)
   a.determinant                    # Korringa–Kohn–Rostoker determinant
   E, I    = a.edc(0.0)             # EDC at k‖ = 0 Å⁻¹
   k, I    = a.mdc(-0.5)            # MDC at E = −0.5 eV
   k, I_up, I_dn = a.edc_spin(0.0) # spin-resolved EDC

**Crystal structure** (from ``*.pot`` via ase2sprkkr ``Potential``):

.. code-block:: python

   c = e.crystal                    # CrystalData
   c.bravais_type                   # 'hexagonal'
   c.point_group                    # 'D_6h'
   c.alat_bohr                      # lattice constant (Bohr)
   c.alat_angstrom                  # lattice constant (Å)
   c.nq                             # number of inequivalent sites
   c.nt                             # number of atom types
   c.type_labels                    # ['W', 'Se', 'Se']

**SCF / potential** (from ``*.pot`` and ``*.inp`` via ase2sprkkr):

.. code-block:: python

   s = e.scf                        # SCFData
   s.fermi_energy_ev                # Fermi energy (eV)
   s.fermi_energy_ry                # Fermi energy (Ry)
   s.irel                           # 1=non-rel, 2=scalar, 3=full-relativistic
   s.fullpot                        # True = full-potential, False = ASA
   s.nspin                          # 1 = non-magnetic, 2 = magnetic
   s.xc_potential                   # 'VWN'
   s.scf_iterations                 # number of SCF cycles
   s.rmsavv                         # RMS charge-density change (convergence)

   r, V = e.get_radial_potential('W')   # V(r) for tungsten
   r, rho = e.get_radial_charge('W')    # ρ(r) for tungsten

**Photon source** (from ``*.inp`` and ``*_SPEC.out`` via ase2sprkkr ``SpecResult``):

.. code-block:: python

   ph = e.photon                    # PhotonSourceData
   ph.photon_energy_ev              # 60.0
   ph.theta_inc_deg                 # photon incidence angle
   ph.polarization_label            # 'C+'
   ph.stokes_s0                     # total intensity
   ph.stokes_s3_pct                 # circular polarisation degree (%)
   ph.photon_wavevector             # (3,) array  [Å⁻¹]
   ph.vector_potential_re           # Jones vector Re part
   ph.vector_potential_im           # Jones vector Im part

**LKKR geometry** (from ``in_structur.inp`` via ase2sprkkr ``structure_file_to_atoms``):

.. code-block:: python

   g = e.lkkr_geometry              # LKKRGeometryData
   g.n_layers                       # number of layers in stack
   g.layer_z_positions              # z-coordinates (alat)
   g.semiinfinite_positions_bohr    # (N, 3) Cartesian atom positions (Bohr)
   g.barrier_ibar                   # potential barrier type

**ASE Atoms** (reconstructed from stored data):

.. code-block:: python

   atoms = e.to_ase_atoms()         # semi-infinite slab (default)
   atoms = e.to_ase_atoms(semiinfinite=False)  # bulk unit cell

Step 6 — Visualise
-------------------

.. code-block:: python

   from oscarpes import visualize as viz

   viz.arpes_map(e)               # I(k‖, E) colour map
   viz.spin_polarization(e)       # P(k‖, E) map + P(k‖) cuts
   viz.arpes_overview(e)          # 4-panel: ARPES / spin asymmetry / P / determinant
   viz.edc_stack(e, [0., 0.5, 1.0])  # stacked normalised EDCs
   viz.mdc_stack(e, [-0.1, -0.5])    # stacked normalised MDCs
   viz.arpes_geometry(e)          # photon geometry + Stokes table
   viz.shape_functions(e)         # SFN panel boundaries per mesh
   viz.voronoi_cells(e)           # 3D Voronoi cell isosurfaces
   viz.potential_overview(e)      # 3-panel: RMT/RWS + layer stack + SFN
   viz.semiinfinite_structure(e)  # 2D cross-section (z vs x)

Bundled demo scripts
--------------------

The repository ships with a small Au example under ``examples/ARPES_K`` and
two runnable demo scripts:

.. code-block:: bash

   python examples/visualization_examples.py
   python examples/plot_semiinfinite.py

Both scripts bootstrap a small local demo database from the bundled example
if no entries are present yet.

Step 7 — Filter and batch operations
--------------------------------------

.. code-block:: python

   # Find all C+ entries
   c_plus = db.find(polarization='C+')

   # Find all Se2W entries at 60 eV
   hits = db.find(formula='Se2W', photon_energy_ev=60.)

   # CD-ARPES: find the C− partner automatically
   partner = db.find_cd_partner(e)
   fig = viz.cd_arpes(e, partner)

   # Batch load intensity arrays for ML
   X, y, ids = db.batch_load('intensity_total', formula_contains='Se2W')
   # X.shape = (N, NK, NE)

   # Export to pandas
   df = db.to_dataframe()

   # PyTorch streaming
   ds = db.as_pytorch_dataset(filter="irel = 3")
