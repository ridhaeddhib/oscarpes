"""
oscarpes.visualize
==========================
Publication-quality visualizations for OSCARpes entries.

All functions accept an OSCAREntry and return a matplotlib Figure.

Available plots
---------------
arpes_map(e)                      — ARPES I(k,E) colour map
cd_arpes(e_cplus, e_cminus)       — CD-ARPES ΔI = I(C+)−I(C−), requires two entries
find_cd_partner(entry, db)        — find the C−/C+ partner entry in a database
spin_polarization(e)              — P(k,E) map + P(k) line cuts
arpes_overview(e)                 — 4-panel: ARPES / spin asymmetry / spin pol / determinant
edc_stack(e, k_values)            — multiple normalised EDCs
mdc_stack(e, e_values)            — multiple normalised MDCs
radial_potential(e)               — V(r) per real atom type
radial_charge(e)                  — ρ(r) per real atom type
rmt_rws_spheres(e)                — bar chart + 2D sphere schematic
shape_functions(e)                — SFN panel boundaries per mesh type
arpes_geometry(e)                 — photon geometry diagram + Stokes table
semiinfinite_structure(e)         — 2D semiinfinite cross-section (z vs x)
potential_overview(e)             — 3-panel: RMT/RWS + semiinfinite z + SFN volumes
voronoi_cells(e)                  — 3D Voronoi cell isosurfaces from SFN data

Notes
-----
CD-ARPES (circular dichroism) requires two *separate* calculations that
are identical except for the photon circular polarization (C+ and C−).
It is NOT the spin-channel difference I↑−I↓ within a single calculation.
Use ``find_cd_partner`` to locate the partner entry automatically, or pass
both entries explicitly to ``cd_arpes``.

ARPES data is stored as shape (NE, NK): energy is axis-0, k∥ is axis-1.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


from .entry import OSCAREntry

_BOHR_TO_ANG = 0.529177210903
_CMAP_ARPES  = 'inferno'
_CMAP_CD     = 'RdBu_r'
_CMAP_SPIN   = 'RdBu_r'


def _ef_line(ax, lw=0.8, color='white', ls='--', alpha=0.6):
    ax.axhline(0, color=color, lw=lw, ls=ls, alpha=alpha)


def _extent(a):
    return [a.k_axis[0], a.k_axis[-1], a.energy_axis[0], a.energy_axis[-1]]


def _imshow(ax, data, extent, cmap, vmin=None, vmax=None, **kw):
    """Display a (NE, NK) ARPES map with correct axis orientation.

    SPR-KKR SPC scans can store k descending (+k→−k) and/or energy descending
    (0→−E), which produces a 180° rotated image.  We normalise both axes to
    [k_min, k_max] × [E_min, E_max] and flip the data array to match.
    """
    data = np.asarray(data)
    k0, k1, e0, e1 = extent
    if k0 > k1:                     # k stored high→low: flip columns
        data = data[:, ::-1]
        k0, k1 = k1, k0
    if e0 > e1:                     # energy stored 0→−E: flip rows
        data = data[::-1, :]
        e0, e1 = e1, e0
    return ax.imshow(data, aspect='auto', origin='lower',
                     extent=[k0, k1, e0, e1],
                     cmap=cmap, vmin=vmin, vmax=vmax, **kw)


# ── 1. ARPES map ──────────────────────────────────────────────────────────────

def arpes_map(entry: OSCAREntry, log_scale=False,
              show=False, filename=None, **kw) -> plt.Figure:
    """Single ARPES I(k,E) colour map."""
    a = entry.arpes
    fig, ax = plt.subplots(figsize=(5, 4))
    data = a.intensity_total
    if log_scale:
        data = np.log1p(data / (data.max() * 1e-4))
    vmax = np.percentile(data, 99.5)
    im = _imshow(ax, data, _extent(a), _CMAP_ARPES, vmin=0, vmax=vmax)
    _ef_line(ax)
    ax.set_xlabel(r'$k_\parallel$ (Å$^{-1}$)')
    ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)')
    ax.set_title(fr'{entry.formula}  |  $h\nu={entry.photon.photon_energy_ev:.0f}$ eV  |  '
                 fr'{entry.photon.polarization_label}')
    plt.colorbar(im, ax=ax, label='arb. units')
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 2. CD-ARPES ──────────────────────────────────────────────────────────────

def find_cd_partner(entry: OSCAREntry, db) -> Optional['OSCAREntry']:
    """
    Find the circular-dichroism partner of *entry* in *db*.

    A partner must match on formula, photon energy, and theta_inc, but have
    the opposite sign of the Stokes s3 parameter (i.e. C+ ↔ C−).

    Parameters
    ----------
    entry : OSCAREntry
    db    : OSCARDatabase

    Returns
    -------
    OSCAREntry or None
    """
    s3_ref  = entry.photon.stokes_s3_pct
    hv_ref  = entry.photon.photon_energy_ev
    th_ref  = entry.photon.theta_inc_deg
    frm_ref = entry.formula

    for eid in db.entry_ids():
        if eid == entry.entry_id:
            continue
        e = db[eid]
        if (e.formula == frm_ref
                and abs(e.photon.photon_energy_ev - hv_ref) < 0.5
                and abs(e.photon.theta_inc_deg    - th_ref) < 0.5
                and e.photon.stokes_s3_pct * s3_ref < 0):   # opposite sign
            return e
    return None


def cd_arpes(entry_cplus: OSCAREntry, entry_cminus: OSCAREntry,
             normalize: bool = True,
             show: bool = False, filename=None) -> plt.Figure:
    """
    CD-ARPES map ΔI = I(C+) − I(C−) or asymmetry A = ΔI / (I(C+)+I(C−)).

    Requires two separate OSCAR entries — one calculated with left-circular
    (C+) photon polarization and one with right-circular (C−).  This is
    *not* the spin-channel difference within a single calculation.

    Parameters
    ----------
    entry_cplus  : OSCAREntry  — calculation with C+ polarization
    entry_cminus : OSCAREntry  — calculation with C− polarization
    normalize    : bool        — True → asymmetry A,  False → raw ΔI
    """
    a_p = entry_cplus.arpes
    a_m = entry_cminus.arpes

    I_plus  = a_p.intensity_total   # shape (NE, NK)
    I_minus = a_m.intensity_total

    delta = I_plus - I_minus
    if normalize:
        denom = I_plus + I_minus
        with np.errstate(invalid='ignore', divide='ignore'):
            CD = np.where(np.abs(denom) > 1e-15, delta / denom, np.nan)
        label = r'$A = \frac{I_{C+}-I_{C-}}{I_{C+}+I_{C-}}$'
    else:
        CD = delta.astype(float)
        label = r'$\Delta I = I_{C+} - I_{C-}$'

    ext = _extent(a_p)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                              gridspec_kw={'width_ratios': [3, 1]})

    # Main map
    ax  = axes[0]
    lim = np.nanpercentile(np.abs(CD[np.isfinite(CD)]), 99) if np.any(np.isfinite(CD)) else 1.
    im  = _imshow(ax, np.nan_to_num(CD), ext, _CMAP_CD, vmin=-lim, vmax=lim)
    _ef_line(ax, color='gray')
    ax.set_xlabel(r'$k_\parallel$ (Å$^{-1}$)')
    ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)')
    ax.set_title(fr'CD-ARPES  |  {entry_cplus.formula}  |  '
                 fr'$h\nu={entry_cplus.photon.photon_energy_ev:.0f}$ eV')
    plt.colorbar(im, ax=ax, label=label)

    # Energy-integrated CD(k∥)  — average over energy axis (axis=0)
    ax2  = axes[1]
    cd_k = np.nanmean(np.nan_to_num(CD), axis=0)   # shape (NK,)
    ax2.plot(cd_k, a_p.k_axis, color='steelblue', lw=1.5)
    ax2.axvline(0, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel(label)
    ax2.set_ylabel(r'$k_\parallel$ (Å$^{-1}$)')
    ax2.set_title('E-integrated CD')

    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 3. Spin polarization ──────────────────────────────────────────────────────

def spin_polarization(entry: OSCAREntry, e_cuts: Optional[List[float]] = None,
                      show=False, filename=None) -> plt.Figure:
    """P(k,E) map + P(k∥) line cuts at fixed energies."""
    a  = entry.arpes
    P  = a.spin_polarization_masked
    if e_cuts is None:
        e_cuts = [0.0, -0.2, -0.5]
    ext = _extent(a)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Map
    ax = axes[0]
    lim = min(50., float(np.nanpercentile(np.abs(P), 98)))
    im = _imshow(ax, np.nan_to_num(P), ext, _CMAP_SPIN, vmin=-lim, vmax=lim)
    _ef_line(ax, color='k')
    ax.set_xlabel(r'$k_\parallel$ (Å$^{-1}$)')
    ax.set_ylabel(r'$E - E_\mathrm{F}$ (eV)')
    ax.set_title(r'Spin polarization $P$ (%)')
    plt.colorbar(im, ax=ax, label='P (%)')

    # Line cuts at fixed E: P has shape (NE, NK), so P[e_idx, :] gives P(k)
    ax2 = axes[1]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(e_cuts)))
    for e_val, col in zip(e_cuts, colors):
        idx = np.argmin(np.abs(a.energy_axis - e_val))
        P_line = P[idx, :]            # shape (NK,)
        valid  = ~np.isnan(P_line)
        ax2.plot(a.k_axis[valid], P_line[valid], lw=1.5, color=col,
                 label=f'E={a.energy_axis[idx]:.2f} eV')
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel(r'$k_\parallel$ (Å$^{-1}$)')
    ax2.set_ylabel('P (%)')
    ax2.set_title(r'$P(k_\parallel)$ at fixed $E$')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 4. ARPES overview (4-panel) ───────────────────────────────────────────────

def arpes_overview(entry: OSCAREntry, show=False, filename=None) -> plt.Figure:
    """
    4-panel overview: total ARPES / spin asymmetry / spin polarization / determinant.

    All panels use the single entry's data.  CD-ARPES (C+−C−) is a separate
    two-entry operation — see ``cd_arpes`` / ``find_cd_partner``.
    """
    a   = entry.arpes
    ext = _extent(a)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    fig.suptitle(fr'{entry.formula}  $h\nu={entry.photon.photon_energy_ev:.0f}$ eV  '
                 fr'{entry.photon.polarization_label}  '
                 fr'(s3={entry.photon.stokes_s3_pct:.0f}%)',
                 fontsize=13, fontweight='bold')

    # data shape (NE, NK): rows=energy (y), cols=k∥ (x) — no transpose
    kw = dict(aspect='auto', origin='upper', extent=ext)

    # Panel 1: total ARPES intensity
    vmax = np.percentile(a.intensity_total, 99.5)
    im0 = axes[0].imshow(a.intensity_total, cmap=_CMAP_ARPES, vmin=0, vmax=vmax, **kw)
    axes[0].set_title('I total'); axes[0].set_ylabel(r'$E-E_F$ (eV)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Panel 2: spin asymmetry A = (I↑−I↓)/(I↑+I↓) from spin channels
    denom = a.intensity_up + a.intensity_down
    with np.errstate(invalid='ignore', divide='ignore'):
        SA = np.where(np.abs(denom) > 1e-15,
                      (a.intensity_up - a.intensity_down) / denom, np.nan)
    lim  = min(1., float(np.nanpercentile(np.abs(SA[np.isfinite(SA)]), 99))) if np.any(np.isfinite(SA)) else 1.
    im1  = axes[1].imshow(np.nan_to_num(SA), cmap=_CMAP_CD, vmin=-lim, vmax=lim, **kw)
    axes[1].set_title(r'Spin asymmetry $A=(I_\uparrow-I_\downarrow)/(I_\uparrow+I_\downarrow)$')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Panel 3: spin polarization P (%)
    P = np.nan_to_num(a.spin_polarization_masked)
    lim2 = min(50., np.percentile(np.abs(P[P != 0]), 95)) if np.any(P != 0) else 50.
    im2  = axes[2].imshow(P, cmap=_CMAP_SPIN, vmin=-lim2, vmax=lim2, **kw)
    axes[2].set_title('Spin pol. P (%)')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Panel 4: determinant (convergence check — may be absent)
    det = a.determinant
    if det.size == a.intensity_total.size:
        vd  = np.percentile(np.abs(det - 1), 99.5)
        im3 = axes[3].imshow((det - 1), cmap='seismic', vmin=-vd, vmax=vd, **kw)
        axes[3].set_title('det − 1  (convergence)')
    else:
        axes[3].text(0.5, 0.5, 'determinant\nnot available',
                     ha='center', va='center', transform=axes[3].transAxes, color='gray')
        im3 = None
        axes[3].set_title('det − 1  (convergence)')
    if im3 is not None:
        plt.colorbar(im3, ax=axes[3], shrink=0.8)

    for ax in axes:
        ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
        ax.set_xlabel(r'$k_\parallel$ (Å$^{-1}$)')

    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 5 & 6. EDC / MDC stacks ──────────────────────────────────────────────────

def edc_stack(entry: OSCAREntry,
              k_values: Optional[List[float]] = None,
              normalise: bool = True,
              show=False, filename=None) -> plt.Figure:
    """Normalised EDCs at multiple k∥ values."""
    a = entry.arpes
    if k_values is None:
        k_values = np.linspace(a.k_axis[0], a.k_axis[-1], 6).tolist()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(k_values)))
    for k_val, col in zip(k_values, colors):
        E, I = a.edc(k_val)
        k_act = a.k_axis[np.argmin(np.abs(a.k_axis - k_val))]
        if normalise and I.max() > 0:
            I = I / I.max()
        ax.plot(E, I, color=col, lw=1.5, label=fr'$k_\parallel$={k_act:.2f} Å$^{{-1}}$')
    ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel(r'$E - E_\mathrm{F}$ (eV)')
    ax.set_ylabel('I (normalised)' if normalise else 'I (arb.)')
    ax.set_title(f'EDC  |  {entry.formula}')
    ax.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


def mdc_stack(entry: OSCAREntry,
              e_values: Optional[List[float]] = None,
              normalise: bool = True,
              show=False, filename=None) -> plt.Figure:
    """Normalised MDCs at multiple energy values."""
    a = entry.arpes
    if e_values is None:
        e_values = np.linspace(a.energy_axis[0], 0., 6).tolist()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(e_values)))
    for e_val, col in zip(e_values, colors):
        k, I = a.mdc(e_val)
        e_act = a.energy_axis[np.argmin(np.abs(a.energy_axis - e_val))]
        if normalise and I.max() > 0:
            I = I / I.max()
        ax.plot(k, I, color=col, lw=1.5, label=f'E={e_act:.2f} eV')
    ax.set_xlabel(r'$k_\parallel$ (Å$^{-1}$)')
    ax.set_ylabel('I (normalised)' if normalise else 'I (arb.)')
    ax.set_title(f'MDC  |  {entry.formula}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 7. Radial potential ───────────────────────────────────────────────────────

def radial_potential(entry: OSCAREntry, r_max_ang=3.5, log_scale=False,
                     show=False, filename=None) -> plt.Figure:
    """V(r) per real atom type with RMT and RWS markers.

    Parameters
    ----------
    log_scale : bool
        If True, plot ``−V(r)`` on a log y-axis (useful for the deep
        Coulomb region near the nucleus where V(r) < 0).
    """
    labels = entry.real_atom_labels()
    if not labels:
        return plt.figure()
    ncols = len(labels)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 0.8, ncols))

    for ax, lbl, col in zip(axes, labels, colors):
        try:
            g = entry._pot_zarr(f'radial_data/potential/{lbl}')
        except Exception:
            ax.set_title(lbl); continue
        r_bohr = g['r'][:] if 'r' in g and g['r'].size > 0 else None
        V_r = g['V_r'][:] if 'V_r' in g else None
        if r_bohr is None or V_r is None or len(r_bohr) == 0:
            ax.text(0.5, 0.5, 'No radial\ngrid', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(lbl); continue

        # Ensure r and V_r have matching lengths (length mismatch can occur when
        # V_r stores both spin channels, or when the mesh was stored truncated)
        nr = min(len(r_bohr), len(np.asarray(V_r).ravel()))
        r_bohr = np.asarray(r_bohr)[:nr]
        V_r    = np.asarray(V_r).ravel()[:nr]
        r      = r_bohr * _BOHR_TO_ANG

        mask = r <= r_max_ang
        if log_scale:
            # V(r) is negative; plot -V(r) on log scale
            y = -V_r[mask]
            y = np.where(y > 0, y, np.nan)
            ax.semilogy(r[mask], y, color=col, lw=1.2)
            ax.set_ylabel('−V(r) (Ry)  [log]')
        else:
            ax.plot(r[mask], r_bohr[mask] * V_r[mask], color=col, lw=1.2)
            ax.set_ylabel('r·V(r) (Bohr·Ry)')

        rmt = float(g.attrs.get('rmt_bohr', 0.)) * _BOHR_TO_ANG
        rws = float(g.attrs.get('rws_bohr', 0.)) * _BOHR_TO_ANG
        if rmt > 0:
            ax.axvline(rmt, color='steelblue', lw=1., ls='--', label=f'RMT={rmt:.3f} Å')
        if rws > 0:
            ax.axvline(rws, color='firebrick', lw=1., ls=':', label=f'RWS={rws:.3f} Å')

        ax.set_xlabel('r (Å)')
        ax.set_title(f'{lbl}  (Z={g.attrs.get("Z","?")})')
        ax.legend(fontsize=8)

    fig.suptitle(f'Radial Potential  |  {entry.formula}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 8. Radial charge density ──────────────────────────────────────────────────

def radial_charge(entry: OSCAREntry, r_max_ang=3.5,
                  show=False, filename=None) -> plt.Figure:
    """ρ(r) (charge density) per real atom type."""
    labels = entry.real_atom_labels()
    if not labels:
        return plt.figure()
    ncols = len(labels)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 0.8, ncols))

    for ax, lbl, col in zip(axes, labels, colors):
        try:
            g = entry._pot_zarr(f'radial_data/charge/{lbl}')
        except Exception:
            ax.set_title(lbl); continue
        r     = g['r'][:]     if 'r'     in g and g['r'].size > 0 else None
        rho_r = g['rho_r'][:] if 'rho_r' in g else None
        if r is None or rho_r is None or len(r) == 0:
            ax.text(0.5, 0.5, 'No radial\ngrid', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(lbl); continue

        nr    = min(len(r), len(np.asarray(rho_r).ravel()))
        r     = np.asarray(r)[:nr]
        rho_r = np.asarray(rho_r).ravel()[:nr]
        r_ang = r * _BOHR_TO_ANG
        mask  = r_ang <= r_max_ang
        ax.plot(r_ang[mask], r_ang[mask]**2 * rho_r[mask],
                color=col, lw=1.2, label=r'$r^2\rho(r)$')
        ax.set_xlabel('r (Å)')
        ax.set_ylabel(r'$r^2\rho(r)$')
        ax.set_title(f'{lbl}  (Z={g.attrs.get("Z","?")})')
        ax.legend(fontsize=8)

    fig.suptitle(f'Radial Charge Density  |  {entry.formula}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 9. RMT / RWS sphere visualisation ────────────────────────────────────────

def rmt_rws_spheres(entry: OSCAREntry, show=False, filename=None) -> plt.Figure:
    """Bar chart of RMT and RWS radii + 2D sphere schematic along semiinfinite z-axis."""
    mesh_data = [(m['name'] if 'name' in m else f'IM{i:02d}', m)
                 for i, m in enumerate(entry.get_mesh_info())]

    # Collect only real-atom meshes (Z stored in potential group)
    try:
        pg = entry._pot_zarr('radial_data/potential')
        real_info = {k: dict(pg[k].attrs) for k in pg.keys()}
    except Exception:
        real_info = {}

    labels = sorted(real_info.keys())
    rmts   = [real_info[l].get('rmt_bohr', 0.) * _BOHR_TO_ANG for l in labels]
    rwss   = [real_info[l].get('rws_bohr', 0.) * _BOHR_TO_ANG for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: bar chart
    ax = axes[0]
    x  = np.arange(len(labels))
    w  = 0.35
    ax.bar(x - w/2, rmts, w, label='RMT', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, rwss, w, label='RWS', color='firebrick',  alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Radius (Å)')
    ax.set_title('Muffin-tin and Wigner-Seitz radii')
    ax.legend()

    # Panel 2: schematic circles in z-x plane
    ax2  = axes[1]
    z_pos = entry.structure.layer_z_positions
    alat  = entry.structure.alat_angstrom
    if z_pos is not None and alat > 0 and labels:
        z_vals = z_pos * alat
        for iz, z in enumerate(z_vals):
            col = plt.cm.Set2(iz % 8)
            # use first available real-atom RWT
            rmt = (rmts[0] if rmts else 1.0)
            rws = (rwss[0] if rwss else 1.5)
            circ_rmt = mpatches.Circle((0., z), rmt, fill=False,
                                       color=col, lw=1.2, ls='-', label=f'z={z:.2f}Å RMT' if iz==0 else '')
            circ_rws = mpatches.Circle((0., z), rws, fill=False,
                                       color=col, lw=0.8, ls=':', alpha=0.6)
            ax2.add_patch(circ_rmt)
            ax2.add_patch(circ_rws)
        margin = max(rwss) * 1.5 if rwss else 2.
        ax2.set_xlim(-margin, margin)
        ax2.set_ylim(z_vals.min() - margin, z_vals.max() + margin)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x (Å)')
        ax2.set_ylabel('z (Å)')
        ax2.set_title('Spheres along semiinfinite z-axis\n(solid=RMT, dotted=RWS)')
    else:
        ax2.text(0.5, 0.5, 'No semiinfinite geometry\navailable', ha='center',
                 va='center', transform=ax2.transAxes, color='gray')

    fig.suptitle(f'RMT / RWS Spheres  |  {entry.formula}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 10. Shape functions ───────────────────────────────────────────────────────

def shape_functions(entry: OSCAREntry, show=False, filename=None) -> plt.Figure:
    """Shape function SFN(r) per mesh type with panel boundaries.

    Reads the schema written by _write_sfn:
      shape_functions/meshes/IM{idx:02d}/
        attrs: idx, npan, nr, nsfn, rmt_sfn, rmtfill, vol
        jrcut      – panel boundary indices
        sfn_rmesh  – outer radial mesh [Bohr]
        sfn        – shape functions [nsfn, nr]
        sfn_lm     – LM indices
    """
    try:
        sfg = entry._pot_zarr('shape_functions')
    except Exception:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No shape-function data in this entry',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        return fig
    if 'meshes' not in sfg:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'shape_functions/meshes group missing',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        return fig
    mg = sfg['meshes']
    meshes = {}
    for mk in sorted(mg.keys()):
        dg = mg[mk]
        a  = dict(dg.attrs)
        rec = {
            'rmt_sfn':   float(a.get('rmt_sfn', 0.)),
            'rmtfill':   float(a.get('rmtfill', 0.)),
            'vol':       float(a.get('vol', 0.)),
            'npan':      int(a.get('npan', 0)),
            'nr':        int(a.get('nr', 0)),
            'nsfn':      int(a.get('nsfn', 0)),
            'jrcut':     dg['jrcut'][:].tolist()    if 'jrcut'     in dg else [],
            'sfn_rmesh': dg['sfn_rmesh'][:]         if 'sfn_rmesh' in dg else np.array([]),
            'sfn':       dg['sfn'][:]               if 'sfn'       in dg else None,
        }
        meshes[mk] = rec
    rmt_arr     = sfg['rmt_sfn'][:] if 'rmt_sfn' in sfg else None
    rmtfill_arr = sfg['rmtfill'][:] if 'rmtfill' in sfg else None

    nm = len(meshes)
    if nm == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No mesh records found', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return fig

    ncols = min(nm, 4)
    nrows = (nm + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)

    cmap_pan = plt.cm.tab10
    for ai, (mk, rec) in enumerate(meshes.items()):
        ax  = axes[ai // ncols][ai % ncols]
        r   = rec['sfn_rmesh'] * _BOHR_TO_ANG     # → Å
        # ---- Plot SFN values (first component = l=0 if available) ----
        sfn = rec['sfn']
        if sfn is not None and sfn.ndim == 2 and sfn.shape[1] == len(r) and len(r) > 1:
            for i in range(min(sfn.shape[0], 5)):
                ax.plot(r, sfn[i], lw=1.0, alpha=0.8, label=f'lm={i}')
            ax.legend(fontsize=6, loc='upper right')
        elif sfn is not None and sfn.ndim == 1 and len(sfn) == len(r) and len(r) > 1:
            ax.plot(r, sfn, lw=1.0, color='steelblue')
        else:
            # No SFN values — just draw panel boundaries as vertical lines
            ax.set_ylim(0, 1)

        # ---- Panel boundaries (jrcut) ----
        jrcut = rec['jrcut']
        if len(r) > 0 and len(jrcut) > 0:
            for ji, j in enumerate(jrcut):
                if 0 <= j < len(r):
                    ax.axvline(r[j], color=cmap_pan(ji % 10), lw=0.8, ls='--',
                               alpha=0.7, label=f'pan {ji}' if sfn is None else '')

        ax.set_xlabel('r (Å)')
        ax.set_ylabel('SFN')
        ax.set_title(f'{mk}  npan={rec["npan"]}  nsfn={rec["nsfn"]}', fontsize=9)

    # Hide unused axes
    for ai in range(nm, nrows * ncols):
        axes[ai // ncols][ai % ncols].axis('off')

    # Right-side summary: bar chart of rmtfill and vol
    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))
    labels = list(meshes.keys())
    rmt_vals  = [meshes[k]['rmt_sfn']  for k in labels]
    fill_vals = [meshes[k]['rmtfill']  for k in labels]
    vol_vals  = [meshes[k]['vol']      for k in labels]
    x = np.arange(len(labels))
    axes2[0].bar(x - 0.2, rmt_vals,  0.35, label='rmt_sfn',  color='steelblue', alpha=0.8)
    axes2[0].bar(x + 0.2, fill_vals, 0.35, label='rmtfill',  color='firebrick',  alpha=0.8)
    axes2[0].set_xticks(x); axes2[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes2[0].set_ylabel('r (alat)'); axes2[0].set_title('RMT / fill radii'); axes2[0].legend()
    axes2[1].bar(x, vol_vals, color='seagreen', alpha=0.8)
    axes2[1].set_xticks(x); axes2[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes2[1].set_ylabel('vol (alat³)'); axes2[1].set_title('Cell volumes')
    fig2.suptitle(f'Shape Function Radii  |  {entry.formula}', fontsize=11, fontweight='bold')
    plt.tight_layout()

    fig.suptitle(f'Shape Functions SFN(r)  |  {entry.formula}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        fig2.savefig(filename.replace('.', '_radii.', 1) if '.' in str(filename)
                     else str(filename) + '_radii', dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ── 11. ARPES geometry diagram ────────────────────────────────────────────────

def arpes_geometry(entry: OSCAREntry, show=False, filename=None) -> plt.Figure:
    """Photon geometry diagram + Stokes vector table + Jones vector arrows."""
    ph  = entry.photon
    pe  = entry.photoemission
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ---- Geometry sketch ----
    ax = axes[0]
    ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 2); ax.set_aspect('equal')
    ax.axis('off')

    # Surface
    ax.plot([-1.8, 1.8], [0, 0], 'k-', lw=2)
    ax.fill_between([-1.8, 1.8], [0, 0], [-1.5, -1.5], color='lightgray', alpha=0.4)
    ax.text(0, -0.9, 'Crystal', ha='center', va='center', color='gray', fontsize=11)

    # Normal vector
    ax.annotate('', xy=(0, 1.6), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0.1, 1.5, r'$\hat{n}$  (surface normal)', fontsize=9)

    # Photon beam arrow at θ_inc
    theta_r = np.radians(ph.theta_inc_deg)
    dx_ph   = -np.sin(theta_r); dy_ph = -np.cos(theta_r)
    ax.annotate('', xy=(0, 0), xytext=(dx_ph * 1.4, dy_ph * 1.4 + 1.4),
                 arrowprops=dict(arrowstyle='->', color='goldenrod', lw=2))
    ax.text(dx_ph * 0.7 - 0.15, dy_ph * 0.7 + 1.4,
            fr'$h\nu={ph.photon_energy_ev:.0f}$ eV', fontsize=9, color='goldenrod')

    # Photoelectron arrow
    theta_e = np.radians(30.)  # representative
    ax.annotate('', xy=(np.sin(theta_e) * 1.2, np.cos(theta_e) * 1.2),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='steelblue', lw=2))
    ax.text(np.sin(theta_e) * 1.3, np.cos(theta_e) * 1.25 - 0.1,
            r'$e^-$', fontsize=11, color='steelblue')

    # θ arc
    thetas = np.linspace(0, theta_r, 30)
    ax.plot(np.sin(thetas) * 0.4, np.cos(thetas) * 0.4, 'k-', lw=0.8)
    ax.text(np.sin(theta_r / 2) * 0.5, np.cos(theta_r / 2) * 0.5,
            fr'$\theta={ph.theta_inc_deg:.0f}°$', fontsize=8)

    ax.set_title(f'ARPES Geometry  |  {entry.formula}', fontsize=11, fontweight='bold')

    # ---- Stokes table ----
    ax2 = axes[1]
    ax2.axis('off')
    rows = [
        ['Parameter', 'Value', 'Meaning'],
        [r'$S_0$', f'{ph.stokes_s0:.3f}', 'Total intensity'],
        [r'$S_1$ (%)', f'{ph.stokes_s1_pct:.1f}%', 'Linear horiz./vert.'],
        [r'$S_2$ (%)', f'{ph.stokes_s2_pct:.1f}%', 'Linear ±45°'],
        [r'$S_3$ (%)', f'{ph.stokes_s3_pct:.1f}%', 'Circular (−100%=C+)'],
        ['Pol.', ph.polarization_label, ''],
        [r'$\phi_\mathrm{inc}$', f'{ph.phi_inc_deg:.0f}°', 'Azimuth angle'],
    ]
    if ph.photon_wavevector is not None:
        q = ph.photon_wavevector
        rows.append([r'$\mathbf{q}_\mathrm{ph}$',
                     f'[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}]', 'Bohr⁻¹'])
    if ph.vector_potential_re is not None:
        Ar = ph.vector_potential_re
        rows.append(['A Re', f'[{Ar[0]:.2f}, {Ar[1]:.2f}, {Ar[2]:.2f}]', 'Jones vec'])
    if ph.vector_potential_im is not None:
        Ai = ph.vector_potential_im
        rows.append(['A Im', f'[{Ai[0]:.2f}, {Ai[1]:.2f}, {Ai[2]:.2f}]', 'Jones vec'])

    col_labels = rows[0]; table_data = rows[1:]
    tab = ax2.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='left', loc='center',
                    bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('lightgray')
        if r == 0:
            cell.set_facecolor('#d0d8e8')
            cell.set_text_props(fontweight='bold')
    ax2.set_title('Photon parameters', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 12. semiinfinite structure ────────────────────────────────────────────────────────

def _build_semiinf_atoms(g, n_rep: int):
    """
    Build atom positions for the semi-infinite crystal.

    Returns a list of (x_ang, y_ang, z_ang, atype) tuples.

    The stored semiinfinite_* arrays already contain atoms from ALL nlayer layers
    (surface slab). We plot them as-is for k=0, then replicate the bulk-repeat
    unit (atoms from semi_inf_start_layer onward) using the cumulative stacking
    vector for each additional repetition k=1..n_rep.
    """
    a_alat = float(g.alat_2d) if g.alat_2d else 1.   # lattice constant in Å
    # a1_2d / a2_2d are unit vectors; multiply by a_alat to get Å vectors
    a1 = np.array(g.a1_2d if g.a1_2d is not None else [1., 0.]) * a_alat
    a2 = np.array(g.a2_2d if g.a2_2d is not None else [0., 1.]) * a_alat
    if len(a1) == 2: a1 = np.append(a1, 0.)
    if len(a2) == 2: a2 = np.append(a2, 0.)

    atypes  = np.asarray(g.semiinfinite_atype, dtype=int)
    z_alat  = np.asarray(g.semiinfinite_z)
    a1_frac = np.asarray(g.semiinfinite_a1_frac)
    a2_frac = np.asarray(g.semiinfinite_a2_frac)

    # Cumulative bulk stacking vector (sum over bulk-repeat layers)
    si       = int(g.semi_inf_start_layer or 0)
    dz_arr   = np.asarray(g.stacking_dz_arr)  if g.stacking_dz_arr  is not None else np.array([1.])
    da1_arr  = np.asarray(g.stacking_da1_arr) if g.stacking_da1_arr is not None else np.array([0.])
    da2_arr  = np.asarray(g.stacking_da2_arr) if g.stacking_da2_arr is not None else np.array([0.])
    dz_bulk  = float(np.sum(dz_arr[si:]))  * a_alat          # Å per bulk-unit step
    d_bulk   = float(np.sum(da1_arr[si:])) * a1 \
             + float(np.sum(da2_arr[si:])) * a2               # in-plane Å per bulk-unit step

    # k=0: all stored base atoms as-is
    positions = []
    for i in range(len(atypes)):
        r = a1_frac[i] * a1 + a2_frac[i] * a2
        z = float(z_alat[i]) * a_alat
        positions.append((float(r[0]), float(r[1]), z, int(atypes[i])))

    # k=1..n_rep: replicate bulk-unit atoms only (index >= si in the stored array)
    bulk_idx = np.where(np.arange(len(atypes)) >= si)[0]
    for k in range(1, n_rep + 1):
        shift_xy = k * d_bulk
        shift_z  = k * dz_bulk
        for i in bulk_idx:
            r = a1_frac[i] * a1 + a2_frac[i] * a2 + shift_xy
            z = float(z_alat[i]) * a_alat - shift_z
            positions.append((float(r[0]), float(r[1]), z, int(atypes[i])))

def semiinfinite_structure(entry: OSCAREntry,
                           vacuum: float = 10.0, n_bulk: int = 2,
                           scale_radii: float = 0.5,
                           backend: str = 'ase',
                           show: bool = True, filename: Optional[str] = None):
    """Visualise the semi-infinite crystal from ``in_structur.inp``.

    Identical to ``ase2sprkkr show-structure <pot> <in_structur.inp>``.

    Parameters
    ----------
    vacuum : float
        Vacuum height in Å above the slab (default 10, same as show-structure).
    n_bulk : int
        Bulk unit repetitions appended below the slab (default 2).
    scale_radii : float
        Atom radius scale for the ASE viewer (default 0.5, same as show-structure).
    backend : str
        ``'ase'`` (default) — ASE GUI, identical to show-structure.
        ``'plotly'`` — interactive 3-D browser figure.
        ``'matplotlib'`` — 2-D z–x cross-section.
    show : bool
        Open the viewer / browser immediately (default True).
    filename : str, optional
        Write structure to file (CIF/HTML/PNG depending on backend).

    Returns
    -------
    ase.Atoms for ``backend='ase'``, plotly Figure or matplotlib Figure otherwise.
    """
    atoms = _get_semiinfinite_atoms(entry, vacuum=vacuum, n_bulk=n_bulk)
    # When using the ASE backend, also open a bulk window — same as
    # `show-structure -i in_structur.inp` which now shows both panels.
    if backend == 'ase' and show:
        try:
            bulk_atoms = _get_bulk_atoms(entry)
            _dispatch_viz(bulk_atoms, entry, backend='ase',
                          scale_radii=scale_radii, show=True, filename=None)
        except Exception:
            pass
    return _dispatch_viz(atoms, entry, backend=backend,
                         scale_radii=scale_radii, show=show, filename=filename)


def bulk_structure(entry: OSCAREntry,
                   scale_radii: float = 0.5,
                   backend: str = 'ase',
                   show: bool = True, filename: Optional[str] = None):
    """Visualise the bulk unit cell from the potential file.

    Identical to ``ase2sprkkr show-structure <pot>`` (no in_structur.inp).

    Parameters
    ----------
    scale_radii : float
        Atom radius scale for the ASE viewer (default 0.5).
    backend : str
        ``'ase'`` (default), ``'plotly'``, or ``'matplotlib'``.
    show : bool
        Open the viewer / browser immediately (default True).
    filename : str, optional
        Write structure to file.

    Returns
    -------
    ase.Atoms for ``backend='ase'``, plotly Figure or matplotlib Figure otherwise.
    """
    atoms = _get_bulk_atoms(entry)
    return _dispatch_viz(atoms, entry, backend=backend,
                         scale_radii=scale_radii, show=show, filename=filename)


# ── Helpers for semiinfinite_structure / bulk_structure ──────────────────────

def _get_bulk_atoms(entry: OSCAREntry):
    """Return ASE Atoms for the bulk unit cell.

    Path 1: ``potential.atoms`` from the original .pot file.
    Path 2 fallback: reconstruct from stored crystals.zarr data.
    """
    import os
    atoms = None
    paths = entry.get_source_paths()
    pot_path = paths.get('pot')
    if pot_path and os.path.exists(pot_path):
        try:
            from ase2sprkkr.potentials.potentials import Potential
            atoms = Potential.from_file(pot_path).atoms
        except Exception:
            atoms = None
    if atoms is None:
        atoms = entry.to_ase_atoms(semiinfinite=False)
    return atoms

def _get_semiinfinite_atoms(entry: OSCAREntry, vacuum: float = 10.0, n_bulk: int = 2):
    """Return SPRKKRAtoms for the semiinfinite slab.

    Path 1: ``structure_file_to_atoms(in_structur.inp, potential)`` — identical
    to ``ase2sprkkr show-structure``.
    Path 2 fallback: reconstruct from stored Zarr data.
    """
    import os
    atoms = None
    paths = entry.get_source_paths()
    pot_path    = paths.get('pot')
    struct_path = paths.get('in_structur')
    if pot_path and os.path.exists(pot_path) and struct_path and os.path.exists(struct_path):
        try:
            from ase2sprkkr.potentials.potentials import Potential
            from ase2sprkkr.sprkkr.structure import structure_file_to_atoms
            atoms = structure_file_to_atoms(
                struct_path, Potential.from_file(pot_path),
                n_bulk=n_bulk, vacuum_height=vacuum,
            )
        except Exception:
            atoms = None
    if atoms is None:
        atoms = entry.to_ase_atoms(semiinfinite=True)
    return atoms


def _dispatch_viz(atoms, entry, backend: str, scale_radii: float,
                  show: bool, filename: Optional[str]):
    """Route atoms to the requested visualization backend."""
    if backend == 'ase':
        try:
            from ase2sprkkr.ase.visualise import view
        except ImportError:
            from ase.visualize import view
        if filename:
            atoms.write(filename)
        if show:
            view(atoms, scale_radii=scale_radii)
        return atoms
    elif backend == 'plotly':
        return _visualize_semiinfinite_3d(atoms, entry, show=show, filename=filename)
    elif backend == 'matplotlib':
        return _visualize_semiinfinite_2d(atoms, entry, show=show, filename=filename)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Use 'ase', 'plotly', or 'matplotlib'.")


def _visualize_with_x3d(atoms, sphere_size: float = 1.0, show: bool = True):
    """Create X3D visualization for Jupyter notebooks."""
    try:
        import plotly.graph_objects as go
        from IPython.display import HTML
    except ImportError:
        raise ImportError("Plotly and IPython required. Install with: pip install plotly ipython")
    
    # Convert ASE atoms to plotly figure
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    
    # Get element symbols
    from ase.data import chemical_symbols
    symbols = [chemical_symbols[z] for z in numbers]
    
    # Create color map
    unique_symbols = list(set(symbols))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {sym: colors[i % len(colors)] for i, sym in enumerate(unique_symbols)}
    
    traces = []
    for symbol in unique_symbols:
        mask = np.array([s == symbol for s in symbols])
        traces.append(go.Scatter3d(
            x=positions[mask, 0],
            y=positions[mask, 1], 
            z=positions[mask, 2],
            mode='markers',
            name=symbol,
            marker=dict(
                size=6 * sphere_size,
                color=color_map[symbol],
                opacity=0.85,
                line=dict(width=0.5, color='black'),
            ),
            text=[symbol] * np.sum(mask),
            hovertemplate='%{text}<br>x=%{x:.3f} Å<br>y=%{y:.3f} Å<br>z=%{z:.3f} Å<extra></extra>',
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Semi-infinite Structure (X3D)',
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    
    x3d_html = fig.to_html(include_plotlyjs='cdn', div_id="x3d-plot")
    
    if show:
        return HTML(x3d_html)
    
    return x3d_html


def _visualize_semiinfinite_3d(atoms, entry, show: bool = True, filename: Optional[str] = None):
    """Create 3D plotly visualization from ASE atoms."""
    import plotly.graph_objects as go
    
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    
    # Get element symbols
    from ase.data import chemical_symbols
    symbols = [chemical_symbols[z] for z in numbers]
    
    # Create color map
    unique_symbols = list(set(symbols))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {sym: colors[i % len(colors)] for i, sym in enumerate(unique_symbols)}
    
    traces = []
    for symbol in unique_symbols:
        mask = np.array([s == symbol for s in symbols])
        is_vac = symbol.startswith('Vc')
        traces.append(go.Scatter3d(
            x=positions[mask, 0],
            y=positions[mask, 1], 
            z=positions[mask, 2],
            mode='markers',
            name=symbol,
            marker=dict(
                size=6 if not is_vac else 3,
                color=color_map[symbol],
                opacity=0.85 if not is_vac else 0.2,
                line=dict(width=0.5, color='black') if not is_vac else dict(width=0),
            ),
            text=[symbol] * np.sum(mask),
            hovertemplate='%{text}<br>x=%{x:.3f} Å<br>y=%{y:.3f} Å<br>z=%{z:.3f} Å<extra></extra>',
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Semi-infinite structure  |  {entry.formula}',
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    
    if filename:
        fig.write_html(filename)
    if show:
        fig.show()
    return fig


def _visualize_semiinfinite_2d(atoms, entry, show: bool = True, filename: Optional[str] = None):
    """Create 2D matplotlib cross-section from ASE atoms."""
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_xlabel('x (Å)'); ax.set_ylabel('z (Å)')
    ax.set_title(f'Semi-infinite structure  |  {entry.formula}', fontweight='bold')
    
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    
    # Get element symbols
    from ase.data import chemical_symbols
    symbols = [chemical_symbols[z] for z in numbers]
    
    # Create color map
    unique_symbols = list(set(symbols))
    cmap = plt.cm.tab10
    color_map = {sym: cmap(i % 10) for i, sym in enumerate(unique_symbols)}
    
    plotted = set()
    for i, (pos, symbol) in enumerate(zip(positions, symbols)):
        is_vac = symbol.startswith('Vc')
        ax.scatter(pos[0], pos[2], s=100, c=[color_map[symbol]],
                   alpha=0.85 if not is_vac else 0.15,
                   edgecolors='k' if not is_vac else 'none', lw=0.4,
                   label=symbol if symbol not in plotted and not is_vac else '_')
        plotted.add(symbol)
    
    ax.legend(fontsize=9, loc='upper right')
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig

    

# ── 13. Potential overview (3-panel) ─────────────────────────────────────────

def potential_overview(entry: OSCAREntry, show=False, filename=None) -> plt.Figure:
    """3-panel summary: RMT/RWS bar | semiinfinite z-positions | SFN cell volumes."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'Potential Overview  |  {entry.formula}', fontsize=12, fontweight='bold')

    # Panel 1: RMT / RWS bars
    try:
        pg = entry._pot_zarr('radial_data/potential')
        labels = sorted(pg.keys())
        rmts = [float(pg[l].attrs.get('rmt_bohr', 0.)) * _BOHR_TO_ANG for l in labels]
        rwss = [float(pg[l].attrs.get('rws_bohr', 0.)) * _BOHR_TO_ANG for l in labels]
    except Exception:
        labels, rmts, rwss = [], [], []

    ax = axes[0]
    x  = np.arange(len(labels))
    ax.bar(x - 0.2, rmts, 0.4, label='RMT (Å)', color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, rwss, 0.4, label='RWS (Å)', color='firebrick',  alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
    ax.set_ylabel('Radius (Å)'); ax.set_title('Muffin-tin radii')

    # Panel 2: atom z-positions in the base slab (absolute depths)
    # layer_z_positions stores per-layer relative offsets (all 0 in SPR-KKR convention).
    # Instead, build absolute z by adding cumulative stacking offsets per layer.
    ax2 = axes[1]
    geom = entry.lkkr_geometry
    si_z = geom.semiinfinite_z  if geom is not None else None
    alat = float(geom.alat_2d)  if geom is not None and geom.alat_2d else 0.
    if si_z is not None and alat > 0 and geom is not None:
        si_z_arr  = np.asarray(si_z)
        n_atoms   = len(si_z_arr)
        n_lay     = int(geom.n_layers or 1)
        dz_arr    = np.asarray(geom.stacking_dz_arr) if geom.stacking_dz_arr is not None \
                    else np.zeros(n_lay)
        # Distribute atoms evenly across layers to assign cumulative z offset
        atoms_per_layer = max(1, n_atoms // max(n_lay, 1))
        abs_z = np.empty(n_atoms)
        for i in range(n_atoms):
            lay_idx      = min(i // atoms_per_layer, n_lay - 1)
            z_offset_alat = float(np.sum(dz_arr[:lay_idx]))
            abs_z[i]     = (si_z_arr[i] + z_offset_alat) * alat
        ax2.scatter(np.zeros(n_atoms), -abs_z, s=60, c=abs_z,
                    cmap='viridis_r', alpha=0.8, edgecolors='k', lw=0.3)
        ax2.set_xlim(-0.5, 0.5); ax2.set_xticks([])
        ax2.set_ylabel('Depth (Å)'); ax2.set_title('Atom z-positions (base slab)')
        ax2.axhline(0, color='steelblue', ls='--', lw=1, alpha=0.7, label='Surface')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No layer data', ha='center', va='center',
                 transform=ax2.transAxes, color='gray')

    # Panel 3: SFN cell volumes
    ax3 = axes[2]
    try:
        sfg  = entry._pot_zarr('shape_functions')
        vols = sfg['vol'][:] if 'vol' in sfg else None
    except Exception:
        vols = None
    if vols is not None:
        ax3.bar(range(1, len(vols) + 1), vols,
                color=plt.cm.plasma(np.linspace(0.1, 0.9, len(vols))), alpha=0.85)
        ax3.set_xlabel('Site IM'); ax3.set_ylabel('Volume (alat³)')
        ax3.set_title('Voronoi cell volumes')
    else:
        ax3.text(0.5, 0.5, 'No SFN data', ha='center', va='center',
                 transform=ax3.transAxes, color='gray')

    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=200, bbox_inches='tight')
    if show: plt.show()
    return fig


# ── 14. Crystal structure via ase2sprkkr / ASE ───────────────────────────────

def show_structure(entry: OSCAREntry,
                   semiinfinite: bool = True,
                   output: Optional[str] = None,
                   output_format: str = 'cif',
                   vacuum: float = 10.0,
                   n_bulk: int = 2,
                   scale_radii: float = 0.5):
    """
    Visualize the crystal structure of an OSCAREntry.

    Strategy
    --------
    1. If the original .pot (and optionally in_structur.inp) are still on disk
       (paths stored in provenance at ingest time), use ase2sprkkr's native
       ``Potential`` + ``structure_file_to_atoms`` — identical to the
       ``show-structure`` CLI command.
    2. Otherwise fall back to reconstructing an ASE Atoms object from the
       HDF5 data via ``entry.to_ase_atoms(semiinfinite)``.

    In both cases the structure is opened in the ASE GUI (``ase.visualize.view``)
    and, if ``output`` is given, also written to a file.

    Parameters
    ----------
    entry         : OSCAREntry
    semiinfinite          : bool   True = semiinfinite geometry (default), False = bulk unit cell
    output        : str    optional output file path (e.g. 'structure.cif')
    output_format : str    ASE-compatible format string (default 'cif')
    vacuum        : float  vacuum height in Å added above semiinfinite (ase2sprkkr path)
    n_bulk        : int    bulk unit repetitions below semiinfinite (ase2sprkkr path)
    scale_radii   : float  atomic radius scale for ASE viewer (default 0.5)

    Returns
    -------
    ase.Atoms
    """
    import os
    from ase.visualize import view as ase_view

    atoms = None

    # ── path 1: use ase2sprkkr native parsers if files are accessible ─────────
    paths = entry.get_source_paths()
    pot_path    = paths.get('pot')
    struct_path = paths.get('in_structur')

    if pot_path and os.path.exists(pot_path):
        try:
            from ase2sprkkr.potentials.potentials import Potential
            potential = Potential.from_file(pot_path)

            if semiinfinite and struct_path and os.path.exists(struct_path):
                from ase2sprkkr.sprkkr.structure import structure_file_to_atoms
                atoms = structure_file_to_atoms(
                    struct_path, potential,
                    n_bulk=n_bulk, vacuum_height=vacuum,
                )
            else:
                atoms = potential.atoms
        except Exception:
            atoms = None  # fall through to HDF5 path

    # ── path 2: reconstruct from HDF5 ─────────────────────────────────────────
    if atoms is None:
        atoms = entry.to_ase_atoms(semiinfinite=semiinfinite)

    if output:
        from ase.io import write as ase_write
        ase_write(output, atoms, format=output_format)
        print(f'[OSCAR] structure written → {output}')

    ase_view(atoms, viewer='ase', scale_radii=scale_radii)
    return atoms


# ── 14. Voronoi cell isosurfaces ──────────────────────────────────────────────

def voronoi_cells(entry: OSCAREntry,
                  mesh_indices=None,
                  iso: float = 0.5,
                  n_grid: int = 80,
                  show: bool = False,
                  filename: Optional[str] = None) -> plt.Figure:
    """3D Voronoi cell isosurfaces reconstructed from the SFN shape functions.

    The full 3D shape function is reconstructed on a Cartesian grid by
    summing all (lm) components with real spherical harmonics::

        Θ(r) = Σ_lm  sfn_lm(r) × Y_lm_real(θ, φ)

    The Θ = *iso* isosurface (default 0.5) is the Voronoi cell boundary.
    The muffin-tin sphere (cyan wireframe) and Voronoi polyhedron edges
    (from face-vertex data) are overlaid for reference.

    Parameters
    ----------
    entry        : OSCAREntry
    mesh_indices : list of int (1-based) or None → up to 4 unique mesh types
    iso          : float  isosurface level (default 0.5)
    n_grid       : int    Cartesian grid resolution (default 80)
    show         : bool   call plt.show()
    filename     : str    save figure to this path

    Returns
    -------
    matplotlib.figure.Figure
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm

    # ── load SFN from the source file via ase2sprkkr ─────────────────────────
    sfn = None
    paths = entry.get_source_paths()
    pot_path = paths.get('pot')
    sfn_path = paths.get('sfn')
    print(f"[OSCAR] Attempting to load shape function from source files:\n")
    if sfn_path and pot_path:
        import os
        if os.path.exists(sfn_path) and os.path.exists(pot_path):
            try:
                from ase2sprkkr.potentials.potentials import Potential
                from ase2sprkkr.sprkkr.shape_function import read_shape_function
                from ase.units import Bohr
                pot  = Potential.from_file(pot_path)
                alat = float(pot['LATTICE']['ALAT']()) * Bohr
                sfn  = read_shape_function(sfn_path, alat=alat)
                print(f"[OSCAR] Loaded shape function from {sfn_path}")
            except Exception as exc:
                sfn = None

    if sfn is None or len(sfn) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'SFN source file not available for this entry.\n'
                'Re-ingest with the .sfn file present to enable this plot.',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=10, wrap=True)
        ax.axis('off')
        print(f"[OSCAR] SFN data not available for entry {entry.entry_id}. "
              f"Ensure the .sfn file is present at ingest time for this plot.")
        return fig

    # ── select meshes to display ──────────────────────────────────────────────
    if mesh_indices is not None:
        meshes = [sfn.mesh_for_idx(i) for i in mesh_indices if sfn.mesh_for_idx(i)]
        print(f"[OSCAR] Displaying meshes with indices: {mesh_indices}")
    else:
        # deduplicate by (npan, nr) — keep first occurrence, up to 4
        seen, meshes = set(), []
        for m in sfn:
            key = (m.npan, m.nr)
            if key not in seen:
                seen.add(key)
                meshes.append(m)
            if len(meshes) == 4:
                break

    ncols = len(meshes)
    fig = plt.figure(figsize=(5.5 * ncols, 5.5))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        fr'{entry.formula} — Voronoi cell isosurfaces  ($\Theta={iso}$)',
        fontsize=12, color='white')

    try:
        from skimage.measure import marching_cubes as _mc
    except ImportError:
        fig2, ax2 = plt.subplots()
        ax2.text(0.5, 0.5, 'scikit-image required for isosurface rendering.\n'
                 'Install with: pip install scikit-image',
                 ha='center', va='center', transform=ax2.transAxes,
                 color='gray', fontsize=10)
        ax2.axis('off')
        return fig2

    for col, mesh in enumerate(meshes):
        ax = fig.add_subplot(1, ncols, col + 1, projection='3d')

        # reconstruct 3D grid and extract isosurface
        lin, grid = mesh.to_3d_grid(n=n_grid)
        dx = lin[1] - lin[0]
        verts, faces, _, _ = _mc(grid, level=iso, spacing=(dx, dx, dx))
        verts += lin[0]

        # colour faces by z for depth perception
        z_vals     = verts[faces, 2].mean(axis=1)
        face_norm  = mcolors.Normalize(z_vals.min(), z_vals.max())
        face_cols  = cm.plasma(face_norm(z_vals))
        poly = Poly3DCollection(verts[faces], alpha=0.88, linewidth=0.)
        poly.set_facecolor(face_cols)
        ax.add_collection3d(poly)

        # muffin-tin sphere (cyan wireframe)
        u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:15j]
        r = mesh.rmt
        ax.plot_wireframe(r*np.cos(u)*np.sin(v),
                          r*np.sin(u)*np.sin(v),
                          r*np.cos(v),
                          lw=0.4, color='cyan', alpha=0.45, rstride=2, cstride=2)

        # Voronoi face edges (from parsed geometry)
        if mesh.faces:
            alat_val = sfn.alat
            for face in mesh.faces:
                v2 = face['verts'] * alat_val
                xs = np.append(v2[:, 0], v2[0, 0])
                ys = np.append(v2[:, 1], v2[0, 1])
                zs = np.append(v2[:, 2], v2[0, 2])
                ax.plot(xs, ys, zs, 'w-', lw=0.55, alpha=0.55)

        lim = mesh.rmesh[-1] * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel('x (Bohr)', fontsize=7, labelpad=1, color='white')
        ax.set_ylabel('y (Bohr)', fontsize=7, labelpad=1, color='white')
        ax.set_zlabel('z (Bohr)', fontsize=7, labelpad=1, color='white')
        ax.tick_params(labelsize=6, colors='white', pad=0)
        ax.set_facecolor('#111111')
        ax.set_title(
            f'IM{mesh.idx}  npan={mesh.npan}  nsfn={mesh.nsfn}\n'
            f'rmt={mesh.rmt:.3f} Å   nface={mesh.nface}   vol={mesh.vol:.1f} alat³',
            fontsize=8, color='white', pad=4)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#333333')
        ax.view_init(elev=25, azim=40)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if filename:
        fig.savefig(filename, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


