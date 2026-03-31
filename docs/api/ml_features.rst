oscarpes.ml\_features
========================

128-dimensional feature extraction from OSCAREntry objects for machine
learning, similarity search, and dimensionality reduction.

Feature groups
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Indices
     - Description
   * - [0:8]
     - Photon/polarisation scalars (hν, angles, Stokes, work function)
   * - [8:16]
     - SCF convergence (E_F, irel, fullpot, Lloyd, log-convergence, iterations, V_MTZ)
   * - [16:24]
     - k-grid geometry (k range, NK, NE, E range, IMV, surface site)
   * - [24:40]
     - ARPES intensity statistics (log percentiles, Fermi map, centroids)
   * - [40:56]
     - CD-ARPES descriptors (integrated CD, valley positions, asymmetry)
   * - [56:72]
     - Spin polarisation (P percentiles, Fermi spin texture, k-asymmetry)
   * - [72:96]
     - EDC peak descriptors (energy, intensity, bandwidth at 8 k∥ values)
   * - [96:112]
     - MDC peak descriptors (k∥ position, intensity at 8 energies)
   * - [112:120]
     - Fermi surface (dominant k, weight, width, symmetry)
   * - [120:128]
     - Structural / radial (alat, n_layers, nq, nt, RMT/RWS filling)

.. automodule:: oscarpes.ml_features
   :members:
   :undoc-members: False
   :show-inheritance:
