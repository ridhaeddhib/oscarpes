Installation
============

Requirements
------------

oscarpes requires Python ≥ 3.9.

Core runtime dependencies (installed automatically):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Package
     - Min version
     - Purpose
   * - ``numpy``
     - 1.24
     - Array arithmetic throughout
   * - ``scipy``
     - 1.10
     - Peak finding, integration, k-space transforms
   * - ``matplotlib``
     - 3.7
     - All visualisation functions
   * - ``zarr``
     - 2.16
     - Crystal, geometry and potential pool storage
   * - ``lance``
     - 0.9
     - Columnar metadata and ARPES array store
   * - ``pyarrow``
     - 14
     - Arrow schema and columnar serialisation
   * - ``duckdb``
     - 0.10
     - SQL analytics over Lance
   * - ``fsspec``
     - 2023.6
     - Transparent local / S3 / GCS path handling

Optional extras
---------------

Install one or more extras to unlock additional functionality:

.. code-block:: bash

   pip install "oscarpes[sprkkr]"   # ase2sprkkr parsing backend (required for ingestion)
   pip install "oscarpes[ase]"       # ASE Atoms / structure interface
   pip install "oscarpes[ml]"        # scikit-learn, pandas, PyTorch
   pip install "oscarpes[cloud]"     # s3fs, gcsfs for remote storage
   pip install "oscarpes[nomad]"     # NOMAD-lab archive export
   pip install "oscarpes[docs]"      # Sphinx + RTD theme (build these docs)
   pip install "oscarpes[all]"       # all of the above

The ase2sprkkr parsing backend
------------------------------

The ``[sprkkr]`` extra installs ``ase2sprkkr``, which is the **parsing and
plotting backend** for oscarpes.  It is strictly required for any
ingestion call (``ingest_directory``, ``ingest_tree``) and for all functions
in :mod:`oscarpes.parsers`.

Without ``ase2sprkkr`` you can still open an existing database and read
pre-ingested entries, but you cannot add new SPR-KKR calculations.

``ase2sprkkr`` provides the following components used by oscarpes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ase2sprkkr component
     - Used for
   * - ``ARPESOutputFile.from_file()``
     - Reading ``*_data.spc`` ARPES cross-section files
   * - ``SpecResult``
     - Parsing ``*_SPEC.out`` (Stokes vector, photon wavevector, Jones vector, potential barrier, basis vectors)
   * - ``InputParameters.from_file()``
     - Reading ``*.inp`` ARPES task parameter files
   * - ``Potential.from_file()``
     - Reading ``*.pot`` self-consistent potential files (all SPR-KKR sections)
   * - ``read_shape_function()`` / ``ShapeFunction``
     - Reading ``*.sfn`` shape function files
   * - ``structure_file_to_atoms()``
     - Reading ``in_structur.inp`` into an SPRKKRAtoms object
   * - ``ExponentialMesh``
     - Radial mesh engine for potential and charge extraction
   * - ``SPRKKRAtoms``
     - ASE Atoms subclass with SPR-KKR metadata in ``atoms.info``

Standard installation (editable)
---------------------------------

.. code-block:: bash

   git clone https://github.com/ridhaeddhib/oscarpes
   cd oscarpes
   pip install -e ".[sprkkr,ase,ml]"

Minimal installation (read-only, no ingestion)
-----------------------------------------------

.. code-block:: bash

   pip install oscarpes

In this mode you can open existing databases and run all analysis functions
but cannot ingest new SPR-KKR calculations.

Verifying the installation
--------------------------

.. code-block:: python

   import oscarpes
   # Verify ase2sprkkr backend is available
   from oscarpes.parsers import parse_spc   # ImportError if ase2sprkkr missing

Building the documentation
--------------------------

.. code-block:: bash

   pip install "oscarpes[docs]"
   cd docs
   make html
   # Open _build/html/index.html

This repository uses a ``src/`` layout. The Sphinx config already adds
``../src`` to ``sys.path``, so building the docs from the repo root does
not require a separate editable install.
