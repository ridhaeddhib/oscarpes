"""
oscarpes.postprocess
============================
Post-processing routines for OSCAREntry objects.

All functions operate on the in-memory OSCAREntry and return plain
NumPy arrays or dicts — no I/O, no matplotlib.

Functions
---------
edc_peaks(e, k_val)              EDC peak positions at fixed k∥
mdc_peaks(e, e_val)              MDC peak positions at fixed E
all_edc_peaks(e)                 Peak map across all k∥ values
cd_integrated_k(e)               Energy-integrated CD per k∥
valley_polarization(e)           k∥ of maximum CD magnitude
bandwidth(e, k_val)              Valence band bandwidth from EDC
band_dispersion(e, threshold)    Band dispersion E(k) from MDC peaks
fermi_surface_k(e, e_tol)        Fermi k∥ values from Fermi map
spin_texture(e, e_val)           Spin texture P(k∥) at fixed E
kz_from_hv(hv, theta, work_fn, V0)  k_z via free-electron final state
cd_asymmetry_map(e)              Full A(k,E) map (thresholded)
radial_moments(r, f)             Radial moments ⟨rⁿ⟩ of V(r) or ρ(r)
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import numpy as np

from .entry import OSCAREntry

_BOHR_TO_ANG  = 0.529177210903
_HBAR2_2M_EV  = 3.81  # ℏ²/(2m) in eV·Å²


# ── Spectral peak finding ─────────────────────────────────────────────────────

def _smooth(y: np.ndarray, sigma: int = 2) -> np.ndarray:
    """Simple Gaussian-like smoothing via uniform convolution."""
    w = min(sigma * 3, len(y) // 4)
    if w < 1:
        return y
    kernel = np.exp(-0.5 * (np.arange(-w, w + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode='same')


def _find_peaks(y: np.ndarray, min_height_frac: float = 0.05,
                min_dist: int = 3) -> np.ndarray:
    """Return indices of local maxima in y."""
    from scipy.signal import find_peaks as sp_find_peaks
    threshold = y.max() * min_height_frac
    peaks, _ = sp_find_peaks(y, height=threshold, distance=min_dist)
    return peaks


def edc_peaks(entry: OSCAREntry, k_val: float,
              smooth_sigma: int = 2) -> Dict:
    """
    Find peaks in the EDC at k∥ closest to k_val.

    Returns
    -------
    dict with keys:
      'k_actual'   : float, actual k∥ used
      'E_peaks'    : array of peak energies (eV)
      'I_peaks'    : array of peak intensities
      'E_axis'     : full energy axis
      'I_edc'      : full (smoothed) EDC intensity
    """
    a = entry.arpes
    E, I_raw = a.edc(k_val)
    k_actual  = a.k_axis[np.argmin(np.abs(a.k_axis - k_val))]
    I_smooth  = _smooth(I_raw, smooth_sigma)
    peaks     = _find_peaks(I_smooth)
    return {
        'k_actual': k_actual,
        'E_peaks':  E[peaks],
        'I_peaks':  I_smooth[peaks],
        'E_axis':   E,
        'I_edc':    I_smooth,
    }


def mdc_peaks(entry: OSCAREntry, e_val: float,
              smooth_sigma: int = 2) -> Dict:
    """Find peaks in the MDC at E closest to e_val."""
    a = entry.arpes
    k, I_raw = a.mdc(e_val)
    e_actual  = a.energy_axis[np.argmin(np.abs(a.energy_axis - e_val))]
    I_smooth  = _smooth(I_raw, smooth_sigma)
    peaks     = _find_peaks(I_smooth)
    return {
        'e_actual': e_actual,
        'k_peaks':  k[peaks],
        'I_peaks':  I_smooth[peaks],
        'k_axis':   k,
        'I_mdc':    I_smooth,
    }


def all_edc_peaks(entry: OSCAREntry,
                  smooth_sigma: int = 2,
                  min_height_frac: float = 0.05) -> Dict:
    """
    Find main EDC peak at every k∥ value.

    Returns
    -------
    dict:
      'k_axis'      : (NK,) k∥ values
      'E_peak'      : (NK,) energy of dominant EDC peak per k (NaN if none found)
      'I_peak'      : (NK,) intensity of dominant peak
    """
    a  = entry.arpes
    NK = len(a.k_axis)
    E_peak = np.full(NK, np.nan)
    I_peak = np.zeros(NK)

    for ik in range(NK):
        I_raw = a.intensity_total[:, ik]
        I_s   = _smooth(I_raw, smooth_sigma)
        peaks = _find_peaks(I_s, min_height_frac)
        if len(peaks) > 0:
            best       = peaks[np.argmax(I_s[peaks])]
            E_peak[ik] = a.energy_axis[best]
            I_peak[ik] = I_s[best]

    return {'k_axis': a.k_axis, 'E_peak': E_peak, 'I_peak': I_peak}


def band_dispersion(entry: OSCAREntry,
                    e_min: Optional[float] = None,
                    e_max: float = 0.0,
                    smooth_sigma: int = 2) -> Dict:
    """
    Extract approximate band dispersion E(k∥) from MDC peaks across
    an energy range.

    Returns dict with 'k_axis', 'E_values', 'k_peaks_per_E', 'E_grid'.
    """
    a      = entry.arpes
    e_mask = a.energy_axis <= e_max
    if e_min is not None:
        e_mask &= a.energy_axis >= e_min
    E_grid = a.energy_axis[e_mask]

    all_k_peaks: List[np.ndarray] = []
    for iE in np.where(e_mask)[0]:
        I_raw = a.intensity_total[iE, :]
        I_s   = _smooth(I_raw, smooth_sigma)
        peaks = _find_peaks(I_s)
        all_k_peaks.append(a.k_axis[peaks] if len(peaks) else np.array([]))

    return {'E_grid': E_grid, 'k_peaks_per_E': all_k_peaks,
            'k_axis': a.k_axis}


# ── CD-ARPES post-processing ──────────────────────────────────────────────────

def cd_integrated_k(entry: OSCAREntry,
                    e_min: Optional[float] = None,
                    e_max: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Energy-integrated CD  ∫ΔI(k,E) dE  as a function of k∥.

    Returns (k_axis, cd_k) arrays.
    """
    a    = entry.arpes
    CD   = a.cd_arpes          # (NE, NK)
    e_ax = a.energy_axis
    mask = e_ax <= e_max
    if e_min is not None:
        mask &= e_ax >= e_min
    cd_k = np.trapezoid(CD[mask, :], e_ax[mask], axis=0)
    return a.k_axis, cd_k


def valley_polarization(entry: OSCAREntry) -> Dict:
    """
    Locate the k∥ positions of maximum positive and negative CD asymmetry.
    Relevant for valley polarization in TMDC materials like WSe₂.

    Returns dict with 'k_pos_max', 'k_neg_max', 'A_pos', 'A_neg',
    'A_integrated_k' (energy-integrated asymmetry per k∥).
    """
    a     = entry.arpes
    A     = np.nan_to_num(a.cd_asymmetry)   # (NE, NK)
    A_int = np.trapezoid(A, a.energy_axis, axis=0)  # (NK,)

    i_pos = np.argmax(A_int)
    i_neg = np.argmin(A_int)
    return {
        'k_pos_max':    float(a.k_axis[i_pos]),
        'k_neg_max':    float(a.k_axis[i_neg]),
        'A_pos':        float(A_int[i_pos]),
        'A_neg':        float(A_int[i_neg]),
        'A_integrated_k': A_int,
        'k_axis':       a.k_axis,
    }


def cd_asymmetry_map(entry: OSCAREntry,
                     threshold: float = 1e-12) -> np.ndarray:
    """
    CD asymmetry A(k,E) = (I↑−I↓)/(I↑+I↓), masked to NaN where I_total < threshold.
    Shape (NE, NK).
    """
    a     = entry.arpes
    denom = a.intensity_up + a.intensity_down
    with np.errstate(invalid='ignore', divide='ignore'):
        A = np.where(np.abs(a.intensity_total) > threshold,
                     (a.intensity_up - a.intensity_down) / denom,
                     np.nan)
    return A


# ── Spectroscopic descriptors ─────────────────────────────────────────────────

def bandwidth(entry: OSCAREntry, k_val: float = 0.0,
              smooth_sigma: int = 2) -> float:
    """
    Estimate the valence band width from the EDC at k_val.
    Defined as the energy difference between the topmost peak and
    the half-maximum point on the low-energy side.
    """
    res   = edc_peaks(entry, k_val, smooth_sigma)
    E     = res['E_axis']
    I     = res['I_edc']
    peaks = np.where(~np.isnan(res['E_peaks']))[0]
    if len(res['E_peaks']) == 0:
        return 0.0
    # topmost peak = closest to EF
    top_E  = np.max(res['E_peaks'])
    top_I  = I[np.argmin(np.abs(E - top_E))]
    half   = top_I * 0.5
    below  = np.where((E < top_E) & (I < half))[0]
    if len(below) == 0:
        return float(np.abs(E[-1] - top_E))
    low_E = E[below[0]]
    return float(abs(top_E - low_E))


def fermi_surface_k(entry: OSCAREntry,
                    e_tol: float = 0.05) -> np.ndarray:
    """
    k∥ positions where the Fermi map intensity exceeds 20% of its maximum.
    """
    fm   = entry.arpes.fermi_map(e_tol)
    mask = fm > 0.2 * fm.max()
    return entry.arpes.k_axis[mask]


def spin_texture(entry: OSCAREntry,
                 e_val: float = 0.0) -> Dict:
    """
    Spin texture P(k∥) at fixed energy.
    Returns dict with 'k_axis', 'P_raw', 'P_masked' (NaN where I_total ≈ 0).
    """
    a     = entry.arpes
    e_idx = np.argmin(np.abs(a.energy_axis - e_val))
    P_raw = a.spin_polarization[e_idx, :]
    I_row = a.intensity_total[e_idx, :]
    P_msk = np.where(np.abs(I_row) > 1e-12, P_raw, np.nan)
    return {
        'e_actual': float(a.energy_axis[e_idx]),
        'k_axis':   a.k_axis,
        'P_raw':    P_raw,
        'P_masked': P_msk,
    }


# ── Final-state k_z conversion ────────────────────────────────────────────────

def kz_from_hv(hv: float, theta_deg: float,
               work_fn: float = 4.5, V0: float = 10.0) -> float:
    """
    Compute k_z using the free-electron final state model.

    k_z = (1/ℏ) √(2m (E_kin·cos²θ + V0))
        = √((E_kin·cos²θ + V0) / (ℏ²/2m))

    Parameters
    ----------
    hv       : photon energy [eV]
    theta_deg: emission polar angle [degrees]
    work_fn  : work function [eV]
    V0       : inner potential [eV]

    Returns
    -------
    k_z [Å⁻¹]
    """
    E_kin = hv - work_fn
    theta = np.radians(theta_deg)
    kz2   = (E_kin * np.cos(theta) ** 2 + V0) / _HBAR2_2M_EV
    return float(np.sqrt(max(kz2, 0.)))


def kz_scan(hv_array: np.ndarray, theta_deg: float = 0.,
            work_fn: float = 4.5, V0: float = 10.) -> np.ndarray:
    """k_z values for a photon-energy scan at fixed emission angle."""
    return np.array([kz_from_hv(hv, theta_deg, work_fn, V0) for hv in hv_array])


# ── Radial data moments ───────────────────────────────────────────────────────

def radial_moments(r: np.ndarray, f: np.ndarray,
                   orders: List[int] = [1, 2, 3]) -> Dict[int, float]:
    """
    Compute radial moments ⟨rⁿ⟩ = ∫ r^(n+2) f(r) dr / ∫ r² f(r) dr.

    Useful for characterising the radial extent of V(r) or ρ(r).
    """
    if len(r) == 0 or len(f) == 0 or len(r) != len(f):
        return {n: 0. for n in orders}
    r2f   = r ** 2 * f
    norm  = np.trapezoid(r2f, r)
    if norm == 0.:
        return {n: 0. for n in orders}
    return {n: float(np.trapezoid(r ** n * r2f, r) / norm) for n in orders}


def rmt_filling(entry: OSCAREntry) -> Dict[str, float]:
    """
    Ratio RMT/RWS for each real atom type — a sphere-filling descriptor
    used as an ML feature and convergence check.
    """
    result = {}
    for lbl in entry.real_atom_labels():
        try:
            pg  = entry._pot_zarr(f'radial_data/potential/{lbl}')
            rmt = float(pg.attrs.get('rmt_bohr', 0.))
            rws = float(pg.attrs.get('rws_bohr', 0.))
            result[lbl] = rmt / rws if rws > 0 else 0.
        except Exception:
            pass
    return result


# ── Summary descriptor ────────────────────────────────────────────────────────

def summary(entry: OSCAREntry) -> Dict:
    """
    Compute a comprehensive dict of physical descriptors for one entry.
    Useful for quick inspection and as input to ML pipelines.
    """
    a   = entry.arpes
    ph  = entry.photon
    sc  = entry.scf

    # EDC at Gamma (k=0)
    E0, I_edc0  = a.edc(0.0)
    edc0_peak_E = E0[I_edc0.argmax()] if I_edc0.max() > 0 else np.nan

    # MDC at EF
    k_ef, I_mdc_ef = a.mdc(0.0)
    mdc_ef_k_peak  = k_ef[I_mdc_ef.argmax()] if I_mdc_ef.max() > 0 else np.nan

    # CD
    k_cd, cd_k_int = cd_integrated_k(entry)
    vp              = valley_polarization(entry)

    # Spin at EF
    sp_ef = spin_texture(entry, 0.0)
    P_ef  = sp_ef['P_masked']
    P_max = float(np.nanmax(np.abs(P_ef))) if not np.all(np.isnan(P_ef)) else 0.

    return {
        # Identity
        'formula':           entry.formula,
        'entry_id':          entry.entry_id,

        # Photon
        'photon_energy_ev':  ph.photon_energy_ev,
        'polarization':      ph.polarization_label,
        'stokes_s3_pct':     ph.stokes_s3_pct,
        'theta_inc_deg':     ph.theta_inc_deg,

        # SCF
        'fermi_energy_ev':   sc.fermi_energy_ev,
        'xc_potential':      sc.xc_potential,
        'irel':              sc.irel,
        'scf_iterations':    sc.scf_iterations,
        'rmsavv':            sc.rmsavv,
        'scf_status':        sc.scf_status,

        # ARPES grid
        'NK':                a.NK,
        'NE':                a.NE,
        'k_min':             float(a.k_axis[0]),
        'k_max':             float(a.k_axis[-1]),
        'e_min':             float(a.energy_axis.min()),

        # Key spectral features
        'edc_gamma_peak_ev': float(edc0_peak_E),
        'mdc_ef_peak_k':     float(mdc_ef_k_peak),
        'bandwidth_gamma':   bandwidth(entry, 0.0),
        'fermi_map_max_k':   float(a.k_axis[a.fermi_map().argmax()]),

        # CD
        'cd_integral_sum':   float(np.sum(np.abs(cd_k_int))),
        'valley_k_pos':      vp['k_pos_max'],
        'valley_k_neg':      vp['k_neg_max'],
        'valley_A_pos':      vp['A_pos'],
        'valley_A_neg':      vp['A_neg'],

        # Spin
        'spin_pol_max_ef':   P_max,

        # Structure
        'alat_bohr':         entry.structure.alat_bohr,
        'n_layers':          entry.structure.n_layers,
    }
