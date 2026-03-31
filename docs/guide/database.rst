Database
========

oscarpes uses a **two-tier storage layout** that separates fast metadata
queries from compressed scientific data:

.. code-block:: text

   <db_root>/
     entries.lance/          ← LanceDB table ``entries`` on local filesystems
     entries.zarr/           ← per-entry JSON sidecars
       <entry_id>/
         parsed_metadata/
           input/
           runtime/
           job/
     crystals.zarr/          ← shared bulk crystals pool  (SHA-256 keyed)
       <sha256>/
     lkkr_geometry.zarr/     ← shared LKKR layer geometry pool
       <sha256>/
     potentials.zarr/        ← shared SCF + radial + shape-function pool
       <sha256>/
         scf/                  SCF convergence scalars
         radial_data/          V(r), ρ(r) per atom type
         shape_functions/      SFN data per mesh
         provenance/           file hashes, raw inp content, source paths
         parsed_metadata/
           potential/          full parsed potential JSON sidecar
     nomad/                  ← NOMAD archive JSONs (generated on demand)

Lance tier
----------

The Lance table is named ``entries`` and on local filesystems is typically
materialized as ``entries.lance/``. Each row contains:

* **Identifiers** — ``entry_id``, ``crystal_sha``, ``geom_sha``, ``pot_sha``,
  ``spc_sha256``.
* **Crystal scalars** (denormalised for fast filtering) — ``formula``,
  ``bravais_type``, ``point_group``, ``alat_bohr``, ``nq``, ``nt``, ``n_layers``.
* **SCF scalars** — ``irel``, ``nspin``, ``fullpot``, ``xc_potential``,
  ``fermi_energy_ev``, ``scf_status``.
* **Photon source** — ``photon_energy_ev``, ``polarization``,
  ``theta_inc_deg``, ``phi_inc_deg``, Stokes vector components.
* **Photon geometry vectors** — ``photon_wavevector``, ``jones_vector_re``,
  ``jones_vector_im``, potential barrier parameters.
* **k-path definition** — ``ka`` (origin), ``k1`` (translation vector),
  ``nk1`` (number of k-points) from the SPR-KKR ``.inp`` file.
* **Energy range** — ``emaxev``, ``eminev`` from the SPC file.
* **ARPES arrays** (float32) — ``k_axis``, ``energy_axis``,
  ``intensity_total``, ``intensity_up``, ``intensity_down``,
  ``spin_polarization``, ``determinant``.
* **Promoted runtime/job fields** — ``kkrspec_version``,
  ``execution_datetime``, ``mpi_nprocs``, ``cpu_time_sec``,
  ``wall_time_sec``, ``stop_status``, ``slurm_partition``,
  ``slurm_ntasks``.

Zarr JSON sidecars
------------------

Large or evolving parsed metadata is not flattened into Lance. Instead it is
stored as JSON payloads in Zarr sidecars:

* ``entries.zarr/<entry_id>/parsed_metadata/input`` — parsed ``*.inp`` fields
* ``entries.zarr/<entry_id>/parsed_metadata/runtime`` — parsed main ``*.out``
  metadata
* ``entries.zarr/<entry_id>/parsed_metadata/job`` — parsed batch-script / SLURM
  metadata
* ``potentials.zarr/<pot_sha>/parsed_metadata/potential`` — parsed potential
  sections such as occupation, reference system, charge moments, and
  magnetisation direction

Zarr pools
----------

Shared data is stored **once per unique crystal / geometry / potential**,
identified by SHA-256.  Two calculations on the same crystal structure share
one ``crystals.zarr/<sha>/`` group, regardless of photon parameters.  This
deduplication is essential for large datasets where many calculations use the
same self-consistent potential.

SHA-256 keys:

* ``crystal_sha`` — derived from Bravais type, lattice constant, and atom
  types (Z number + label), so it identifies the bulk crystal topology.
* ``geom_sha`` — SHA-256 of the full ``in_structur.inp`` file.
* ``pot_sha`` — SHA-256 of the ``*.pot_new`` / ``*.pot`` potential file.
* ``spc_sha256`` — SHA-256 of the ``*_data.spc`` file (used for duplicate
  detection during ingestion).

Opening a database
------------------

.. code-block:: python

   from oscarpes import OSCARDatabase

   db = OSCARDatabase('/data/oscar_db/')
   print(db)   # formatted summary of all entries

The ``db_path`` can be a local directory or a cloud URI (``s3://bucket/prefix``,
``gcs://bucket/prefix``) — all access goes through ``fsspec``.

Listing entries
---------------

.. code-block:: python

   rows = db.list_entries()   # list of metadata dicts, no ARPES arrays
   for r in rows:
       print(r['entry_id'], r['formula'], r['photon_energy_ev'])

Pool statistics
---------------

.. code-block:: python

   ps = db.pool_summary()
   # {'entries': 42, 'crystals': 5, 'lkkr_geometry': 12, 'potentials': 15}

Filtering
---------

:meth:`~oscarpes.OSCARDatabase.find` returns a list of
:class:`~oscarpes.entry.OSCAREntry` objects matching the given criteria.
Filtering is pushed down to Lance (columnar predicate) for efficiency.

.. code-block:: python

   # Single condition
   wse2   = db.find(formula='WSe2')
   cplus  = db.find(polarization='C+')
   relat  = db.find(irel=3)                   # full-relativistic only

   # Combined
   hits   = db.find(formula='WSe2', photon_energy_ev=60., polarization='C+')

   # Photon energy with tolerance (default ±2 eV)
   hits   = db.find(photon_energy_ev=60., photon_energy_tol=1.)

   # Extra keyword args — any scalar column
   hits   = db.find(nspin=2)                  # magnetic calculations
   hits   = db.find(xc_potential='VWN')

Finding CD-ARPES partners
--------------------------

CD-ARPES requires two calculations that are identical except for the circular
polarisation (C+ and C−).  The database can find the partner automatically:

.. code-block:: python

   e_cplus = db['osc-WSe2-1']
   e_cminus = db.find_cd_partner(e_cplus)   # matches crystal_sha + opposite polarization

   if e_cminus:
       from oscarpes import visualize as viz
       fig = viz.cd_arpes(e_cplus, e_cminus)

Batch loading for ML
--------------------

.. code-block:: python

   # Load all intensity_total arrays as a 3D numpy array
   X, y, ids = db.batch_load('intensity_total', formula_contains='WSe2')
   # X.shape = (N, NK, NE),  y = photon energies,  ids = entry_id list

   # Other datasets: 'intensity_up', 'intensity_down', 'spin_polarization',
   #                 'cd_arpes' (computed on the fly as up − down)
   X, _, _ = db.batch_load('cd_arpes', label_attr='polarization')

Pandas export
-------------

.. code-block:: python

   df = db.to_dataframe()
   # Returns a pandas DataFrame with all scalar metadata columns (no arrays)
   # Useful for statistics, groupby, and external analysis tools.

PyTorch streaming
-----------------

.. code-block:: python

   ds     = db.as_pytorch_dataset(filter="irel = 3 AND nspin = 2")
   loader = torch.utils.data.DataLoader(ds, batch_size=32)
   for batch in loader:
       ids  = batch['entry_id']
       X    = torch.tensor(batch['intensity_total'])

Deleting entries
----------------

:meth:`~oscarpes.OSCARDatabase.delete` removes rows from the Lance ``entries``
table.
It does **not** remove Zarr pool data (which may be shared by other entries).

.. code-block:: python

   db.delete(['osc-WSe2-1', 'osc-WSe2-2'])

   # Raise KeyError if any ID is not found
   db.delete(['osc-WSe2-99'], raise_if_missing=True)

Inspecting the Zarr tree
-------------------------

.. code-block:: python

   # Summary of all pools
   db.print_tree()

   # Detailed tree for one entry
   db.print_tree('osc-WSe2-1', max_depth=4)

Entry IDs
---------

Entry IDs follow the scheme ``osc-<Hill-formula>-<n>`` (e.g. ``osc-Se2W-1``).
The Hill formula is derived automatically from the potential file using
``SPRKKRAtoms.get_chemical_formula('hill')`` from ase2sprkkr.  The counter
``<n>`` is incremented from the highest existing ID for that formula, so
uniqueness is preserved even after deletions.

Database location
-----------------

When calling :meth:`~oscarpes.entry.OSCAREntry.from_directory` without an
explicit ``db_path``, the default database location is ``~/.oscarpes/``.
For scripts, it is usually better to use an explicit absolute path:

.. code-block:: python

   import os
   _HERE = os.path.dirname(os.path.abspath(__file__))
   DB_PATH = '/data/oscar_db'

   eid = ingest_directory(calc_dir, DB_PATH, formula='WSe2')
