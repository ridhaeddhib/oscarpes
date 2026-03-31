oscarpes.ingest
==================

Ingest SPR-KKR ARPES calculations into the oscarpes v3 database.

The ingestion pipeline uses the **ase2sprkkr** parsing backend for all
file reading:

1. :func:`~oscarpes.parsers.find_files` discovers all files in the directory.
2. ``parse_spc`` (``ARPESOutputFile``) reads the ARPES cross sections.
3. ``parse_spec_out`` (``SpecResult``) reads Stokes vector and photon geometry.
4. ``parse_inp`` (``InputParameters``) reads ARPES task parameters.
5. ``parse_pot`` (``Potential``) reads the self-consistent potential.
6. ``parse_sfn`` (``read_shape_function``) reads the shape function.
7. ``structure_file_to_atoms`` reads the LKKR layer geometry.

Shared data (crystal / geometry / potential) is deduplicated by SHA-256 and
written once to the Zarr pools. ARPES arrays and stable query-oriented
metadata are written to Lance; richer parsed metadata is stored as JSON
sidecars in Zarr.

Storage layout
--------------

.. code-block:: text

   <db_root>/
     entries.lance/              # LanceDB table ``entries`` on local disk
     entries.zarr/<entry_id>/
       parsed_metadata/{input,runtime,job}/
     crystals.zarr/<crystal_sha>/
     lkkr_geometry.zarr/<geom_sha>/
     potentials.zarr/<pot_sha>/
       scf/
       radial_data/
       shape_functions/
       provenance/
       parsed_metadata/potential/
     nomad/

.. automodule:: oscarpes.ingest
   :members:
   :undoc-members: False
   :show-inheritance:
