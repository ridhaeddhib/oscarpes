Visualisation
=============

:mod:`oscarpes.visualize` provides 16 publication-quality plot functions.
Every function accepts an :class:`~oscarpes.entry.OSCAREntry` as its first
argument and returns a :class:`matplotlib.figure.Figure`.

All plot functions import matplotlib and use ase2sprkkr's data objects
(``ShapeFunction``, ``ShapeFunctionMesh``) for Voronoi / shape-function plots.

ARPES intensity maps
--------------------

.. code-block:: python

   from oscarpes import visualize as viz

   # Total ARPES cross section I(k‖, E) — colour map with colour bar
   fig = viz.arpes_map(e)

   # 4-panel overview: I(k‖, E) / spin asymmetry / P(k‖, E) / determinant
   fig = viz.arpes_overview(e)

Spin polarisation
-----------------

.. code-block:: python

   # P(k‖, E) colour map plus integrated P(k‖) line cuts at several energies
   fig = viz.spin_polarization(e)

Circular dichroism (CD-ARPES)
------------------------------

CD-ARPES shows the asymmetry ΔI = I(C+) - I(C-) between two calculations
with opposite circular polarisation.  It requires **two separate entries**:

.. code-block:: python

   # Manual: pass both entries explicitly
   fig = viz.cd_arpes(e_cplus, e_cminus)

   # Automatic: find the partner from the database
   partner = db.find_cd_partner(e)
   fig = viz.cd_arpes(e, partner)

   # Find the partner without the database object
   from oscarpes.visualize import find_cd_partner
   partner = find_cd_partner(e, db)

.. note::
   CD-ARPES is the photon helicity asymmetry ``I(C+) - I(C-)``.  It is
   **not** the spin-channel difference ``I↑ - I↓`` within a single entry.

EDC and MDC stacks
-------------------

.. code-block:: python

   # Stack of normalised energy distribution curves at selected k∥ values
   fig = viz.edc_stack(e, k_values=[-1.0, -0.5, 0.0, 0.5, 1.0])

   # Stack of normalised momentum distribution curves at selected energies
   fig = viz.mdc_stack(e, e_values=[-0.1, -0.3, -0.5, -0.8, -1.0])

Photon geometry
---------------

.. code-block:: python

   # Sketch of photon incidence geometry + table of Stokes parameters
   # Stokes data comes from *_SPEC.out parsed by ase2sprkkr SpecResult
   fig = viz.arpes_geometry(e)

Semi-infinite structure
-----------------------

.. code-block:: python

   # 2D cross-section of the semi-infinite slab (z vs x)
   # Layer positions come from in_structur.inp via structure_file_to_atoms
   fig = viz.semiinfinite_structure(e)

Radial potential and charge
----------------------------

.. code-block:: python

   # V(r) for each real atom type (from potentials.zarr radial_data)
   fig = viz.radial_potential(e)

   # ρ(r) for each real atom type
   fig = viz.radial_charge(e)

Muffin-tin / Wigner–Seitz spheres
-----------------------------------

.. code-block:: python

   # Bar chart of RMT and RWS radii + 2D sphere schematic
   fig = viz.rmt_rws_spheres(e)

Shape functions
---------------

The shape function is read from ``*.sfn`` by ase2sprkkr's
``read_shape_function()`` / ``ShapeFunction``.

.. code-block:: python

   # Panel boundaries of the shape function per mesh type
   fig = viz.shape_functions(e)

Voronoi cells
-------------

3D isosurface visualisation of the Voronoi / full-potential cells using the
``ShapeFunction`` and ``ShapeFunctionMesh`` objects from ase2sprkkr:

.. code-block:: python

   fig = viz.voronoi_cells(e)
   fig = viz.voronoi_cells(e, n_grid=30)   # higher resolution grid

Potential overview
------------------

.. code-block:: python

   # 3-panel composite: RMT/RWS radii + semi-infinite layer stack + SFN volumes
   fig = viz.potential_overview(e)

Saving figures
--------------

Every function accepts ``filename`` and ``dpi`` keyword arguments:

.. code-block:: python

   viz.arpes_map(e, filename='arpes.pdf')
   viz.arpes_map(e, filename='arpes.png', dpi=300)

All 16 functions at a glance
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``arpes_map(e)``
     - ARPES I(k,E) colour map
   * - ``cd_arpes(e_cplus, e_cminus)``
     - CD-ARPES ΔI = I(C+) - I(C-)
   * - ``find_cd_partner(entry, db)``
     - Find the C-/C+ partner entry in a database
   * - ``spin_polarization(e)``
     - P(k,E) map + P(k) line cuts
   * - ``arpes_overview(e)``
     - 4-panel: ARPES / spin asymmetry / spin pol / determinant
   * - ``edc_stack(e, k_values)``
     - Multiple normalised EDCs
   * - ``mdc_stack(e, e_values)``
     - Multiple normalised MDCs
   * - ``radial_potential(e)``
     - V(r) per real atom type
   * - ``radial_charge(e)``
     - ρ(r) per real atom type
   * - ``rmt_rws_spheres(e)``
     - Bar chart + 2D sphere schematic
   * - ``shape_functions(e)``
     - SFN panel boundaries per mesh type
   * - ``arpes_geometry(e)``
     - Photon geometry diagram + Stokes table
   * - ``semiinfinite_structure(e)``
     - 2D semi-infinite cross-section (z vs x)
   * - ``potential_overview(e)``
     - 3-panel: RMT/RWS + semi-infinite z + SFN volumes
   * - ``voronoi_cells(e)``
     - 3D Voronoi cell isosurfaces from SFN data
