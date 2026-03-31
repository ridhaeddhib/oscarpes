Post-processing
===============

:mod:`oscarpes.postprocess` provides 17 analysis functions that operate
on in-memory :class:`~oscarpes.entry.OSCAREntry` objects and return plain
NumPy arrays or Python dicts.  No I/O, no matplotlib.

All functions use ``scipy.signal.find_peaks`` for peak detection and
``numpy.trapezoid`` for integration.

.. code-block:: python

   from oscarpes import postprocess as pp

EDC peak finding
----------------

.. code-block:: python

   result = pp.edc_peaks(e, k_val=0.0)
   # result keys:
   #   'k_actual'  : float   — actual k∥ used (nearest grid point)
   #   'E_peaks'   : array   — energies of peaks (eV)
   #   'I_peaks'   : array   — intensities at peaks
   #   'E_axis'    : array   — full energy axis
   #   'I_edc'     : array   — smoothed EDC

   # Control smoothing
   result = pp.edc_peaks(e, k_val=0.5, smooth_sigma=3)

MDC peak finding
----------------

.. code-block:: python

   result = pp.mdc_peaks(e, e_val=-0.5)
   # result keys:
   #   'e_actual'  : float   — actual energy used
   #   'k_peaks'   : array   — k∥ positions of peaks (Å⁻¹)
   #   'I_peaks'   : array   — intensities at peaks
   #   'k_axis'    : array   — full k∥ axis
   #   'I_mdc'     : array   — smoothed MDC

Peak map across all k∥
-----------------------

.. code-block:: python

   result = pp.all_edc_peaks(e)
   # result keys:
   #   'k_axis'  : (NK,) k∥ values
   #   'E_peak'  : (NK,) energy of dominant peak per k (NaN if none found)
   #   'I_peak'  : (NK,) intensity of dominant peak

Band dispersion E(k)
--------------------

Extract the approximate band dispersion by finding MDC peaks across a
range of energies:

.. code-block:: python

   result = pp.band_dispersion(e, e_min=-1.5, e_max=0.0)
   # result keys:
   #   'E_grid'          : energies used
   #   'k_peaks_per_E'   : list of arrays, one per energy
   #   'k_axis'          : full k∥ axis

CD-ARPES analysis
-----------------

**Energy-integrated CD** ΔI integrated over energy as a function of k∥:

.. code-block:: python

   k, cd_k = pp.cd_integrated_k(e)              # full energy range
   k, cd_k = pp.cd_integrated_k(e, e_min=-0.5)  # restrict to E > −0.5 eV

**CD asymmetry map** A(k,E) = (I↑ − I↓)/(I↑ + I↓), masked where I_total ≈ 0:

.. code-block:: python

   A = pp.cd_asymmetry_map(e)          # (NE, NK) array, NaN where I < threshold
   A = pp.cd_asymmetry_map(e, threshold=1e-10)

**Valley polarisation** (maximum CD asymmetry positions, relevant for TMDCs):

.. code-block:: python

   vp = pp.valley_polarization(e)
   # vp keys:
   #   'k_pos_max'      : k∥ of maximum positive CD (Å⁻¹)
   #   'k_neg_max'      : k∥ of maximum negative CD (Å⁻¹)
   #   'A_pos'          : value at k_pos_max
   #   'A_neg'          : value at k_neg_max
   #   'A_integrated_k' : (NK,) energy-integrated asymmetry
   #   'k_axis'         : full k∥ axis

Bandwidth
---------

Valence band width from the EDC at a given k∥, defined as the energy
difference between the topmost peak and the half-maximum point on the
low-energy side:

.. code-block:: python

   bw = pp.bandwidth(e, k_val=0.0)     # bandwidth in eV at k‖ = 0
   bw = pp.bandwidth(e, k_val=0.5)     # at k‖ = 0.5 Å⁻¹

Fermi surface k-positions
--------------------------

.. code-block:: python

   k_f = pp.fermi_surface_k(e)             # default e_tol=0.05 eV
   k_f = pp.fermi_surface_k(e, e_tol=0.1)

Spin texture
------------

.. code-block:: python

   st = pp.spin_texture(e, e_val=0.0)  # at E_F
   # st keys:
   #   'e_actual'  : actual energy used
   #   'k_axis'    : (NK,) k∥ values
   #   'P_raw'     : (NK,) raw polarisation P(k∥)
   #   'P_masked'  : (NK,) P with NaN where I_total < 1e-12

k_z from photon energy
-----------------------

Free-electron final-state k_z conversion using:
k_z = √((E_kin·cos²θ + V0) / (ℏ²/2m)):

.. code-block:: python

   kz = pp.kz_from_hv(hv=60., theta_deg=0., work_fn=4.5, V0=10.)

   # Photon-energy scan (k_z dispersion)
   import numpy as np
   hv_range = np.arange(40., 100., 2.)
   kz_arr   = pp.kz_scan(hv_range, theta_deg=0., work_fn=4.5, V0=10.)

Radial moments
--------------

Compute radial moments ⟨rⁿ⟩ = ∫ r^(n+2) f(r) dr / ∫ r² f(r) dr of any
radial function (useful for V(r) or ρ(r)):

.. code-block:: python

   r, V = e.get_radial_potential('W')
   moments = pp.radial_moments(r, V, orders=[1, 2, 3])
   # {'1': ⟨r⟩, '2': ⟨r²⟩, '3': ⟨r³⟩}

RMT/RWS sphere filling ratio
------------------------------

.. code-block:: python

   filling = pp.rmt_filling(e)
   # {'W': 0.82, 'Se': 0.78}  — RMT/RWS ratio per atom type

Comprehensive summary dict
---------------------------

:func:`~oscarpes.postprocess.summary` computes all key descriptors in one call:

.. code-block:: python

   desc = pp.summary(e)
   # Keys include:
   #   formula, entry_id
   #   photon_energy_ev, polarization, stokes_s3_pct, theta_inc_deg
   #   fermi_energy_ev, xc_potential, irel, scf_iterations, rmsavv, scf_status
   #   NK, NE, k_min, k_max, e_min
   #   edc_gamma_peak_ev, mdc_ef_peak_k, bandwidth_gamma, fermi_map_max_k
   #   cd_integral_sum, valley_k_pos, valley_k_neg, valley_A_pos, valley_A_neg
   #   spin_pol_max_ef
   #   alat_bohr, n_layers

Complete function reference
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function
     - Returns
   * - ``edc_peaks(e, k_val)``
     - Dict: E_peaks, I_peaks, smoothed EDC
   * - ``mdc_peaks(e, e_val)``
     - Dict: k_peaks, I_peaks, smoothed MDC
   * - ``all_edc_peaks(e)``
     - Dict: E_peak(k), I_peak(k)
   * - ``band_dispersion(e, e_min, e_max)``
     - Dict: E_grid, k_peaks_per_E
   * - ``cd_integrated_k(e, e_min, e_max)``
     - Tuple: (k_axis, cd_k)
   * - ``valley_polarization(e)``
     - Dict: k_pos_max, k_neg_max, A_pos, A_neg
   * - ``cd_asymmetry_map(e, threshold)``
     - ndarray (NE, NK): A(k,E)
   * - ``bandwidth(e, k_val)``
     - float: bandwidth in eV
   * - ``fermi_surface_k(e, e_tol)``
     - ndarray: k∥ at Fermi level
   * - ``spin_texture(e, e_val)``
     - Dict: P_raw(k), P_masked(k)
   * - ``kz_from_hv(hv, theta_deg, work_fn, V0)``
     - float: k_z in Å⁻¹
   * - ``kz_scan(hv_array, theta_deg, work_fn, V0)``
     - ndarray: k_z for each hv
   * - ``radial_moments(r, f, orders)``
     - Dict: {n: ⟨rⁿ⟩}
   * - ``rmt_filling(e)``
     - Dict: {label: RMT/RWS}
   * - ``summary(e)``
     - Dict: comprehensive physical descriptors
