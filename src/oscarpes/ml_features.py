"""
oscarpes.ml_features
============================
Feature extraction for machine learning from OSCAREntry objects.

The ``extract_features`` function returns a fixed-length float64 vector
(128 dimensions by default) encoding all physically meaningful aspects
of one ARPES calculation: photon parameters, structure, SCF state,
spectral shape, CD dichroism, spin texture, and radial potential info.

All features are normalised to be dimensionless or in natural units so
they can be fed directly into sklearn / PyTorch pipelines.

Functions
---------
extract_features(entry)    128-dim feature vector for one entry
feature_names()            list of 128 human-readable feature names
batch_extract(entries)     (N, 128) array for a list of entries
feature_dataframe(entries) pandas DataFrame (if pandas is available)
"""
from __future__ import annotations
from typing import List, Optional
import warnings
import numpy as np

from .entry     import OSCAREntry
from .postprocess import (
    edc_peaks, mdc_peaks, cd_integrated_k, valley_polarization,
    spin_texture, bandwidth, all_edc_peaks, cd_asymmetry_map,
    radial_moments, rmt_filling,
)

_N_FEATURES = 128


def _safe(val, default: float = 0.) -> float:
    """Return ``val`` as float, replacing NaN/Inf/errors with ``default``."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError, IndexError):
        return default


def extract_features(entry: OSCAREntry) -> np.ndarray:
    """
    Extract a 128-dimensional feature vector from one OSCAREntry.

    Feature groups
    --------------
    [0:8]    Photon / polarization scalars
    [8:16]   SCF convergence and electronic structure
    [16:24]  k-grid geometry
    [24:40]  ARPES intensity statistics (percentiles, moments)
    [40:56]  CD-ARPES descriptors (integrated, asymmetry, valley)
    [56:72]  Spin polarization descriptors
    [72:96]  EDC peak positions and widths across k∥
    [96:112] MDC peak positions across energy grid
    [112:120] Fermi surface descriptors
    [120:128] Radial / structural descriptors

    All features are float64. NaN/Inf are replaced by 0.
    """
    feats = np.zeros(_N_FEATURES, dtype=np.float64)
    a     = entry.arpes
    ph    = entry.photon
    pe    = entry.photoemission
    sc    = entry.scf

    # Warn early if intensity data looks corrupt — features will still be finite
    # (sanitised at the end) but the user should know.
    if a.NK > 0 and a.NE > 0:
        finite_frac = np.isfinite(a.intensity_total).mean()
        if finite_frac < 1.0:
            warnings.warn(
                f"[extract_features] entry {entry.entry_id!r}: "
                f"{(1 - finite_frac) * 100:.1f}% of intensity_total values are "
                "NaN/Inf — feature vector may be unreliable.",
                UserWarning, stacklevel=2,
            )

    # ── [0:8] Photon / polarization ───────────────────────────────────────────
    feats[0] = ph.photon_energy_ev
    feats[1] = ph.theta_inc_deg
    feats[2] = ph.phi_inc_deg
    feats[3] = ph.stokes_s0
    feats[4] = ph.stokes_s1_pct / 100.
    feats[5] = ph.stokes_s2_pct / 100.
    feats[6] = ph.stokes_s3_pct / 100.    # −1 = C+, +1 = C−
    feats[7] = pe.work_function_ev

    # ── [8:16] SCF ─────────────────────────────────────────────────────────────
    feats[8]  = sc.fermi_energy_ev
    feats[9]  = sc.irel                   # 1/2/3
    feats[10] = float(sc.fullpot)
    feats[11] = float(sc.lloyd)
    feats[12] = np.log10(max(sc.rmsavv, 1e-30))   # convergence quality
    feats[13] = np.log10(max(sc.rmsavb, 1e-30))
    feats[14] = sc.scf_iterations
    feats[15] = sc.vmtz_ry

    # ── [16:24] k-grid ─────────────────────────────────────────────────────────
    feats[16] = _safe(a.k_axis[0]  if len(a.k_axis) else 0.)
    feats[17] = _safe(a.k_axis[-1] if len(a.k_axis) else 0.)
    feats[18] = float(a.NK)
    feats[19] = float(a.NE)
    feats[20] = _safe(a.energy_axis.min() if len(a.energy_axis) else 0.)
    feats[21] = _safe(a.energy_axis.max() if len(a.energy_axis) else 0.)
    feats[22] = _safe(pe.imv_final_ev)
    feats[23] = _safe(pe.iq_at_surf)

    # ── [24:40] ARPES intensity statistics ─────────────────────────────────────
    I   = a.intensity_total
    Ifl = I.ravel()
    Ifl = Ifl[np.isfinite(Ifl) & (Ifl > 0)]
    if len(Ifl) > 10:
        feats[24] = np.log10(Ifl.max())
        feats[25] = np.log10(np.percentile(Ifl, 99))
        feats[26] = np.log10(np.percentile(Ifl, 90))
        feats[27] = np.log10(np.median(Ifl))
        feats[28] = np.log10(Ifl.mean())
        feats[29] = float(np.sum(Ifl > np.percentile(Ifl, 95))) / len(Ifl)  # hot fraction
    else:
        feats[24:30] = 0.
    # Fermi-map features
    fm = a.fermi_map()
    if fm.max() > 0:
        feats[30] = a.k_axis[fm.argmax()]           # k of Fermi map peak
        feats[31] = float(np.sum(fm > 0.2 * fm.max())) / len(fm)  # Fermi k-extent
        feats[32] = float(np.trapezoid(fm, a.k_axis))   # total Fermi weight
    # Intensity centroid in (k, E)
    I_norm = I / (I.sum() + 1e-30)
    E_grid, K_grid = np.meshgrid(a.energy_axis, a.k_axis, indexing='ij')
    feats[33] = float(np.sum(E_grid * I_norm))   # energy centroid
    feats[34] = float(np.sum(K_grid * I_norm))   # k centroid
    # Intensity asymmetry left vs right in k
    k_mid = (a.k_axis[0] + a.k_axis[-1]) / 2.
    left  = I[:, a.k_axis < k_mid].sum()
    right = I[:, a.k_axis >= k_mid].sum()
    feats[35] = float((right - left) / (right + left + 1e-30))
    feats[36:40] = 0.   # reserved

    # ── [40:56] CD-ARPES descriptors ──────────────────────────────────────────
    CD   = a.intensity_up - a.intensity_down   # (NE, NK)  circular dichroism
    A    = np.nan_to_num(cd_asymmetry_map(entry))  # (NE, NK)
    k_cd, cd_k = cd_integrated_k(entry)
    vp   = valley_polarization(entry)

    feats[40] = float(np.trapezoid(np.abs(cd_k), a.k_axis))  # total |CD| integrated over k
    feats[41] = float(vp['k_pos_max'])
    feats[42] = float(vp['k_neg_max'])
    feats[43] = float(vp['A_pos'])
    feats[44] = float(vp['A_neg'])
    feats[45] = float(abs(vp['k_pos_max'] - vp['k_neg_max']))  # valley separation
    feats[46] = float(abs(vp['A_pos'] + vp['A_neg']))           # symmetry of valley pol
    feats[47] = float(np.nanpercentile(A, 99)) if np.any(np.isfinite(A)) else 0.
    feats[48] = float(np.nanpercentile(A, 1))  if np.any(np.isfinite(A)) else 0.
    feats[49] = float(np.nanmean(np.abs(A)))   if np.any(np.isfinite(A)) else 0.
    # CD energy distribution: how concentrated near EF?
    cd_ef   = np.abs(CD[np.abs(a.energy_axis) < 0.2, :]).sum()
    cd_deep = np.abs(CD[a.energy_axis < -0.5, :]).sum()
    feats[50] = float(cd_ef / (cd_deep + cd_ef + 1e-30))
    feats[51:56] = 0.   # reserved

    # ── [56:72] Spin polarization ─────────────────────────────────────────────
    P_msk = a.spin_polarization_masked  # (NE, NK)
    valid = P_msk[np.isfinite(P_msk)]
    if len(valid) > 10:
        feats[56] = float(np.nanpercentile(P_msk, 99))   # max P
        feats[57] = float(np.nanpercentile(P_msk, 1))    # min P
        feats[58] = float(np.nanmean(np.abs(P_msk)))     # mean |P|
        feats[59] = float(np.nanstd(P_msk))              # std P
    # Spin texture at EF
    sp_ef = spin_texture(entry, 0.0)
    P_ef  = sp_ef['P_masked']
    valid_ef = P_ef[np.isfinite(P_ef)]
    if len(valid_ef) > 5:
        feats[60] = float(np.nanmax(np.abs(P_ef)))
        feats[61] = float(np.nanmean(P_ef))
        # k-asymmetry of P at EF
        k_mid = (a.k_axis[0] + a.k_axis[-1]) / 2.
        P_l = np.nanmean(P_ef[a.k_axis < k_mid])
        P_r = np.nanmean(P_ef[a.k_axis >= k_mid])
        feats[62] = float(P_l - P_r)
    feats[63:72] = 0.   # reserved

    # ── [72:96] EDC peak descriptors ──────────────────────────────────────────
    # Sample 8 k∥ values and record peak E and I
    k_sample = np.linspace(a.k_axis[0], a.k_axis[-1], 8)
    for i, k_val in enumerate(k_sample):
        res = edc_peaks(entry, k_val)
        if len(res['E_peaks']) > 0:
            feats[72 + i]     = float(res['E_peaks'][np.argmax(res['I_peaks'])])   # peak E
            feats[80 + i]     = float(np.log10(max(res['I_peaks'].max(), 1e-30)))  # peak I
        # bandwidth
        bw = bandwidth(entry, k_val)
        feats[88 + min(i, 7)] = bw

    # ── [96:112] MDC peak descriptors ─────────────────────────────────────────
    e_min = _safe(a.energy_axis.min() if len(a.energy_axis) else 0.)
    e_max = _safe(a.energy_axis.max() if len(a.energy_axis) else 0.)
    e_sample = np.linspace(e_min, e_max, 8)
    for i, e_val in enumerate(e_sample):
        res = mdc_peaks(entry, e_val)
        if len(res['k_peaks']) > 0:
            feats[96 + i]  = float(res['k_peaks'][np.argmax(res['I_peaks'])])   # peak k
            feats[104 + i] = float(np.log10(max(res['I_peaks'].max(), 1e-30)))  # peak I

    # ── [112:120] Fermi surface ────────────────────────────────────────────────
    fm = a.fermi_map()
    fm_k = a.k_axis
    if fm.max() > 0:
        feats[112] = float(fm_k[fm.argmax()])             # dominant Fermi k
        feats[113] = float(fm.max())
        feats[114] = float(np.sum(fm > 0.5 * fm.max())) / len(fm)  # width
        feats[115] = float(np.trapezoid(fm / fm.max(), fm_k)) # normalised Fermi weight
        # Symmetry of Fermi surface
        fm_left  = fm[fm_k < fm_k.mean()]
        fm_right = fm[fm_k >= fm_k.mean()]
        sz = min(len(fm_left), len(fm_right))
        if sz > 5:
            corr = float(np.corrcoef(fm_left[:sz], fm_right[-sz:][::-1])[0,1])
            feats[116] = corr if np.isfinite(corr) else 0.
    feats[117:120] = 0.   # reserved

    # ── [120:128] Radial / structural ─────────────────────────────────────────
    feats[120] = entry.structure.alat_bohr
    feats[121] = float(entry.structure.n_layers)
    feats[122] = float(entry.structure.nq)
    feats[123] = float(entry.structure.nt)

    # RMT/RWS filling ratios
    filling = rmt_filling(entry)
    vals    = list(filling.values())
    if vals:
        feats[124] = float(np.mean(vals))
        feats[125] = float(np.std(vals))
    feats[126:128] = 0.   # reserved

    # Sanitise
    feats = np.nan_to_num(feats, nan=0., posinf=0., neginf=0.)
    return feats.astype(np.float64)


def feature_names() -> List[str]:
    """Return human-readable names for all 128 features."""
    names = [
        # [0:8] Photon
        'photon_energy_ev', 'theta_inc_deg', 'phi_inc_deg',
        'stokes_s0', 'stokes_s1_frac', 'stokes_s2_frac', 'stokes_s3_frac',
        'work_function_ev',
        # [8:16] SCF
        'fermi_energy_ev', 'irel', 'fullpot', 'lloyd',
        'log10_rmsavv', 'log10_rmsavb', 'scf_iterations', 'vmtz_ry',
        # [16:24] k-grid
        'k_min', 'k_max', 'NK', 'NE', 'e_min', 'e_max',
        'imv_final_ev', 'iq_at_surf',
        # [24:40] ARPES stats
        'log10_I_max', 'log10_I_p99', 'log10_I_p90', 'log10_I_median',
        'log10_I_mean', 'hot_fraction', 'fermi_k_peak', 'fermi_k_extent',
        'fermi_weight', 'E_centroid', 'k_centroid', 'I_k_asymmetry',
        'reserved_36', 'reserved_37', 'reserved_38', 'reserved_39',
        # [40:56] CD
        'cd_integral', 'valley_k_pos', 'valley_k_neg', 'A_pos', 'A_neg',
        'valley_separation', 'valley_symmetry', 'A_p99', 'A_p01', 'A_mean_abs',
        'cd_ef_fraction', 'reserved_51', 'reserved_52', 'reserved_53',
        'reserved_54', 'reserved_55',
        # [56:72] Spin
        'P_p99', 'P_p01', 'P_mean_abs', 'P_std',
        'P_ef_max_abs', 'P_ef_mean', 'P_ef_k_asymmetry',
        'reserved_63', 'reserved_64', 'reserved_65', 'reserved_66',
        'reserved_67', 'reserved_68', 'reserved_69', 'reserved_70',
        'reserved_71',
    ]
    # [72:80] EDC peak energies at 8 k-values
    names += [f'edc_peak_E_k{i}' for i in range(8)]
    # [80:88] EDC peak intensities
    names += [f'edc_peak_I_k{i}' for i in range(8)]
    # [88:96] EDC bandwidths
    names += [f'edc_bandwidth_k{i}' for i in range(8)]
    # [96:104] MDC peak k at 8 energies
    names += [f'mdc_peak_k_e{i}' for i in range(8)]
    # [104:112] MDC peak intensities
    names += [f'mdc_peak_I_e{i}' for i in range(8)]
    # [112:120] Fermi surface
    names += ['fermi_dominant_k', 'fermi_max_I', 'fermi_width', 'fermi_norm_weight',
              'fermi_symmetry', 'reserved_117', 'reserved_118', 'reserved_119']
    # [120:128] Structure
    names += ['alat_bohr', 'n_layers', 'nq', 'nt',
              'rmt_rws_mean', 'rmt_rws_std', 'reserved_126', 'reserved_127']

    assert len(names) == _N_FEATURES, f'{len(names)} != {_N_FEATURES}'
    return names


def batch_extract(entries: list) -> np.ndarray:
    """
    Extract features from a list of OSCAREntry objects.

    Returns
    -------
    X : np.ndarray  shape (N, 128)
    """
    return np.stack([extract_features(e) for e in entries], axis=0)


def feature_dataframe(entries: list):
    """
    Return a pandas DataFrame with one row per entry and 128 feature columns.
    Requires pandas.
    """
    import pandas as pd
    X    = batch_extract(entries)
    cols = feature_names()
    df   = pd.DataFrame(X, columns=cols)
    df.insert(0, 'entry_id', [e.entry_id for e in entries])
    df.insert(1, 'formula',  [e.formula  for e in entries])
    return df
