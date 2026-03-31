oscarpes.entry
=================

Data model for one OSCARpes v3 database entry.

Data is loaded from two complementary stores:

* **Lance** (table ``entries``, typically ``entries.lance/`` on local disk)
  — metadata scalars and ARPES arrays (fast columnar access).
* **Zarr pools** — scientific data keyed by SHA-256:

  - ``crystals.zarr/<sha>/``        → :class:`CrystalData`
  - ``lkkr_geometry.zarr/<sha>/``   → :class:`LKKRGeometryData`
  - ``potentials.zarr/<sha>/``      → :class:`SCFData` + radial data + shape functions
  - ``entries.zarr/<entry_id>/``    → parsed JSON sidecars for input/runtime/job

All raw SPR-KKR file parsing happens during ingestion via the **ase2sprkkr**
backend.  An ``OSCAREntry`` object never reads SPR-KKR files directly.

.. automodule:: oscarpes.entry
   :members:
   :undoc-members: False
   :show-inheritance:
