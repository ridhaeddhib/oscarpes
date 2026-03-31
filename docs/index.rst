OSCARpes v3 — ARPES Database and Analysis
=============================================

**OSCARpes** (*One-Step Computed Angle-Resolved PhotoEmission Spectroscopy*)
is a Python package for ingesting, storing, querying and
analysing SPR-KKR first-principles ARPES calculations.  It is built on
top of `ase2sprkkr <https://ase2sprkkr.github.io>`_ which provides all
low-level SPR-KKR file parsers, the radial-mesh engine, the shape-function
reader, and the ASE atoms interface.

.. rubric:: What is it?

SPR-KKR (*Spin-Polarised Relativistic Korringa–Kohn–Rostoker*) is a
multiple-scattering code for calculating relativistic electronic structure
and ARPES cross sections from first principles.  A typical workflow
produces:

* a self-consistent potential file (``*.pot``),
* a photoemission cross-section file (``*_data.spc``),
* a geometry file (``in_structur.inp``),
* a SPEC output file (``*_SPEC.out``), and
* a shape-function file (``*.sfn``).

OSCARpes ingests these raw outputs into a **Lance + Zarr** database and
exposes them through a clean Python API.

Database layout at a glance
---------------------------

OSCARpes separates query-oriented row data from pooled scientific data:

.. code-block:: text

   <db_root>/
     entries.lance/          ← LanceDB table ``entries`` on local filesystems
     entries.zarr/           ← per-entry JSON sidecars (input / runtime / job)
     crystals.zarr/          ← shared bulk crystal pool
     lkkr_geometry.zarr/     ← shared semi-infinite geometry pool
     potentials.zarr/        ← shared SCF / radial / SFN / parsed potential pool
     nomad/

.. rubric:: Core dependencies

* **ase2sprkkr** — parsing backend for *all* SPR-KKR file formats and
  the ASE/SPRKKRAtoms interface.  Every call to ``parse_spc``,
  ``parse_spec_out``, ``parse_inp``, ``parse_pot``, ``parse_sfn`` and
  ``structure_file_to_atoms`` delegates to ase2sprkkr internally.
* **Lance / lancedb** — columnar metadata store; supports SQL filtering,
  vector search, and PyTorch streaming without loading full arrays.
* **Zarr** — compressed pool storage for crystals, LKKR layer geometries,
  potentials, radial data, and shape functions.

.. rubric:: Key features

* One-call ingestion of complete SPR-KKR calculation directories.
* Deduplication by SHA-256: shared crystals and potentials are stored only once.
* SQL-style metadata queries over thousands of entries.
* 16 publication-quality visualisation functions built on matplotlib.
* 17 post-processing routines (EDC/MDC fitting, symmetrisation, CD-ARPES,
  spin texture, band mapping, …).
* 128-dimensional ML feature vectors for similarity search and classification.
* NOMAD/FAIRmat archive export plugin.
* Transparent cloud storage via fsspec (S3, GCS).

Quick start
-----------

.. code-block:: python

   from oscarpes import OSCARDatabase
   from oscarpes.ingest import ingest_directory

   # Parse a finished SPR-KKR calculation and store it
   eid = ingest_directory('/path/to/calc_dir', formula='2H-WSe2')  # → ~/.oscarpes/

   # Open the database
   db = OSCARDatabase()  # → ~/.oscarpes/
   print(db)           # summary of all entries

   # Load one entry — all data already in memory
   e = db[eid]
   print(e)

   # ARPES intensity map I(k‖, E)
   from oscarpes import visualize as viz
   fig = viz.arpes_map(e)
   fig.savefig('arpes.pdf')

   # EDC at k‖ = 0 Å⁻¹
   E, I = e.arpes.edc(0.0)

   # Filter: all WSe2 entries at 50 eV with C+ polarisation
   entries = db.find(formula='WSe2', photon_energy_ev=50., polarization='C+')

   # ML feature matrix
   from oscarpes.ml_features import batch_extract
   X = batch_extract(entries)   # shape (N, 128)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   guide/quickstart
   guide/database
   guide/entry
   guide/visualize
   guide/postprocess
   guide/ml_features
   guide/nomad

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/entry
   api/database
   api/ingest
   api/parsers
   api/store
   api/visualize
   api/postprocess
   api/ml_features
   api/nomad_export

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
