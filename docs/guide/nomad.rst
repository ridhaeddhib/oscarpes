NOMAD / FAIRmat export
======================

oscarpes ships a `NOMAD <https://nomad-lab.eu>`_ plugin that registers
a custom parser and schema for the NOMAD research data infrastructure,
enabling FAIR (Findable, Accessible, Interoperable, Reusable) publication
of SPR-KKR ARPES calculations.

Installation
------------

.. code-block:: bash

   pip install "oscarpes[nomad]"

This installs ``nomad-lab`` and ``nomad-simulations`` as additional
dependencies.  The plugin entry points are declared in ``pyproject.toml``
and auto-discovered by NOMAD:

.. code-block:: toml

   [project.entry-points."nomad.plugin"]
   sprkkr_parser = "oscarpes.nomad_plugin:sprkkr_parser"
   sprkkr_schema = "oscarpes.nomad_plugin:sprkkr_schema"

Exporting from the database
-----------------------------

.. code-block:: python

   from oscarpes.nomad_export import export_entry, export_database

   # Export one entry to a NOMAD archive JSON
   export_entry(e, output_dir='nomad_uploads/')

   # Export all entries in a database
   export_database(db, output_dir='nomad_uploads/')

The exported JSON files follow the NOMAD metainfo schema and include:

* Crystal structure (Bravais type, lattice constant, atom types)
* SCF parameters (irel, XC functional, Fermi energy)
* Photon source (energy, polarisation, Stokes vector)
* ARPES cross sections (intensity arrays, energy/k axes)
* Provenance (file hashes, code version, run metadata)

Using the NOMAD parser directly
---------------------------------

When ``nomad-lab`` is installed, the plugin is auto-discovered and can be
invoked from the command line:

.. code-block:: bash

   nomad parse test/ARPES_K/kkrspec.inp

Or programmatically:

.. code-block:: python

   from oscarpes.nomad_plugin import sprkkr_parser
   result = sprkkr_parser.parse('/path/to/calc_dir/kkrspec.inp')

NOMAD metainfo schema
---------------------

The custom schema (``sprkkr_schema``) maps oscarpes data classes to
NOMAD metainfo sections:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - oscarpes object
     - NOMAD section
   * - ``CrystalData``
     - ``System`` → crystal structure
   * - ``SCFData``
     - ``Method`` → DFT / KKR parameters
   * - ``PhotonSourceData``
     - ``Photon`` → photon source
   * - ``ARPESData``
     - ``Outputs`` → spectral properties
   * - Zarr provenance
     - ``Provenance`` → file hashes and metadata
