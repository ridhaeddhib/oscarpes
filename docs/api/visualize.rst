oscarpes.visualize
=====================

Publication-quality visualisations for OSCARpes entries.

All 16 functions accept an :class:`~oscarpes.entry.OSCAREntry` and return
a :class:`matplotlib.figure.Figure`.

Shape-function and Voronoi plots use ase2sprkkr's ``ShapeFunction`` and
``ShapeFunctionMesh`` objects stored in the Zarr potential pool.

.. note::

   CD-ARPES (``cd_arpes``) requires **two separate entries** calculated
   with opposite circular polarisation (C+ and C−).  It is not the
   spin-channel difference within a single entry.  Use
   :func:`find_cd_partner` or
   :meth:`~oscarpes.database.OSCARDatabase.find_cd_partner` to locate
   the partner automatically.

.. automodule:: oscarpes.visualize
   :members:
   :undoc-members: False
   :show-inheritance:
