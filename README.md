# OSCARpes

**One-Step Computed Angle-Resolved PhotoEmission Spectroscopy**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/oscarpes.svg)](https://pypi.org/project/oscarpes/)

A database of *theoretical* photoemission spectra obtained within the one-step
model of photoemission, combining relativistic multiple-scattering theory
(KKR) with Green's-function formalism and full experimental realism
(photon source, sample orientation, detector geometry, ...).

OSCARpes ingests raw [SPR-KKR](https://www.ebert.cup.uni-muenchen.de/old/index.php/en/software/sprkkr)
output files into a **Lance + Zarr** database and exposes them through
a clean Python API for querying, visualisation, post-processing, and
machine learning. All low-level file parsing is handled by
[ase2sprkkr](https://ase2sprkkr.github.io).

---

## Features

| Category | Capability |
|---|---|
| **Ingestion** | One-call ingestion of complete SPR-KKR calculation directories; recursive tree walker for batch import |
| **Deduplication** | SHA-256 content-addressed pools -- shared crystals, geometries, and potentials are stored exactly once |
| **Querying** | SQL-style metadata filtering over thousands of entries via Lance; filter by formula, photon energy, polarization, relativity level, XC functional, etc. |
| **Visualisation** | 17 publication-quality plotting functions (ARPES maps, CD-ARPES, spin polarization, EDC/MDC stacks, radial potentials, shape functions, Voronoi cells, semiinfinite structures, photon geometry diagrams, ...) |
| **Post-processing** | 15 analysis routines -- EDC/MDC peak finding, band dispersion, CD asymmetry, valley polarization, spin texture, Fermi surface, k_z mapping, radial moments, ... |
| **ML** | 128-dimensional feature vectors encoding photon parameters, crystal structure, SCF state, spectral shape, and spin texture; batch extraction to NumPy/pandas; PyTorch `IterableDataset` streaming directly from Lance |
| **NOMAD/FAIRmat** | UNDERDEVELOPMENT!!! Archive export plugin conforming to NOMAD metainfo schema; first KKR photoemission parser in the FAIRmat ecosystem |
| **Cloud** | Transparent S3/GCS storage via fsspec -- same API for local and remote databases |

---

## Installation

```bash
pip install oscarpes
```

Optional extras:

```bash
pip install oscarpes[sprkkr]   # ase2sprkkr parsing backend (required for ingestion)
pip install oscarpes[ml]       # PyTorch + scikit-learn for ML pipelines
pip install oscarpes[nomad]    # NOMAD/FAIRmat archive export
pip install oscarpes[cloud]    # S3 and GCS remote storage
pip install oscarpes[all]      # everything above
pip install oscarpes[dev]      # development (pytest, Jupyter, Sphinx docs)
```

---

## Quick start

### Ingest a SPR-KKR calculation

```python
from oscarpes.ingest import ingest_directory

eid = ingest_directory(
    '/path/to/calc_dir',   # folder with *.inp, *_data.spc, *.pot, etc.
    'oscar_db/',           # database root (created automatically)
    formula='2H-WSe2',
)
print(eid)  # 'osc-2H-WSe2-1'
```

### Batch ingest an entire folder tree

```python
from oscarpes.ingest import ingest_tree

eids = ingest_tree('/path/to/all_calculations', 'oscar_db/')
print(f'{len(eids)} entries ingested')
```

### Open the database and query

```python
from oscarpes import OSCARDatabase

db = OSCARDatabase('oscar_db/')
print(db)  # tabular summary of all entries

# Filter by metadata
entries = db.find(formula='WSe2', photon_energy_ev=50., polarization='C+')

# Access a single entry by ID
e = db['osc-2H-WSe2-1']
print(e)
```

### Explore an entry

```python
# Crystal structure
e.crystal.bravais_type      # 'hexagonal'
e.crystal.alat_angstrom     # lattice constant in Angstrom
e.crystal.point_group       # 'D_6h'

# ARPES data
e.arpes.intensity_total     # np.ndarray shape (NE, NK)
e.arpes.energy_axis         # energy grid in eV
e.arpes.k_axis              # k-parallel grid in 1/Angstrom
E, I = e.arpes.edc(0.0)     # EDC at k=0

# SCF / potential
e.scf.fermi_energy_ev
e.scf.xc_potential          # 'VWN', 'PBE', ...

# Photon source
e.photon.photon_energy_ev
e.photon.polarization_label  # 'C+', 'C-', 'LH', 'LV'
e.photon.stokes_s3_pct       # Stokes S3 (circular polarization %)
```

### Visualise

```python
from oscarpes import visualize as viz

viz.arpes_map(e)                        # ARPES I(k,E) colour map
viz.spin_polarization(e)                # P(k,E) map with line cuts
viz.arpes_overview(e)                   # 4-panel overview
viz.edc_stack(e, k_values=[0, 0.5, 1])  # EDC waterfall
viz.radial_potential(e)                 # V(r) for each atom type
viz.semiinfinite_structure(e)           # layer geometry cross-section
viz.arpes_geometry(e)                   # photon geometry + Stokes table

# Circular dichroism (needs C+ and C- pair)
partner = viz.find_cd_partner(e, db)
viz.cd_arpes(e, partner)
```

### Post-processing

```python
from oscarpes import postprocess as pp

peaks = pp.edc_peaks(e, k_val=0.0)       # EDC peak positions at k=0
disp  = pp.band_dispersion(e)            # E(k) from MDC peaks
cd    = pp.cd_asymmetry_map(e)           # CD asymmetry A(k,E)
sp    = pp.spin_texture(e, e_val=-1.0)   # spin polarization vs k at E=-1 eV
kz    = pp.kz_from_hv(50., theta_deg=0.) # k_z from photon energy
info  = pp.summary(e)                    # full metadata dict
```

### Machine learning

```python
from oscarpes.ml_features import extract_features, batch_extract

# 128-dim feature vector for one entry
fv = extract_features(e)

# Batch: (N, 128) array for all entries
entries = db.find(formula='WSe2')
X = batch_extract(entries)

# PyTorch streaming directly from Lance
ds = db.as_pytorch_dataset(filter="irel = 3")
loader = torch.utils.data.DataLoader(ds, batch_size=32)
```

### NOMAD / FAIRmat export

```python
from oscarpes.nomad_export import export_entry, export_database

export_entry(e, 'wse2_arpes.zip')
export_database(db, 'oscar_upload.zip')
```

---

## Database layout

```
oscar_db/
  entries.lance/           Lance columnar store (one row per spectrum)
  crystals.zarr/           Zarr pool: bulk crystal identity (keyed by SHA-256)
  lkkr_geometry.zarr/      Zarr pool: semi-infinite LKKR layer stacks
  potentials.zarr/         Zarr pool: SCF convergence, radial V(r)/rho(r), shape functions
    <pot_sha>/
      scf/                 SCF metadata and convergence
      radial_data/         V(r), rho(r) per atom type on exponential meshes
      shape_functions/     SFN boundary coefficients per mesh type
      provenance/          calculation provenance (code version, task type)
  nomad/                   NOMAD archive JSONs (FAIRmat export)
  entry_ids.json           ID counter for Materials-Project-style IDs (osc-<formula>-<n>)
```

---

## SPR-KKR file types handled

| File pattern | Parser | Content |
|---|---|---|
| `*.inp` | `parse_inp` | ARPES task parameters (k-path, photon, detector) |
| `*_data.spc` | `parse_spc` | ARPES cross-section arrays I(k,E) |
| `*_SPEC.out` | `parse_spec_out` | Stokes vector, photon geometry, surface barrier |
| `*.pot` / `*_pot_new` | `parse_pot` | Self-consistent potential, radial data |
| `*.sfn` | `parse_sfn` | Shape function boundary coefficients |
| `in_structur.inp` | `structure_file_to_atoms` | LKKR semiinfinite layer geometry |

All parsers delegate to `ase2sprkkr` internally.

---

## Dependencies

**Core** (installed automatically):

`numpy`, `scipy`, `matplotlib`, `zarr` (v3), `lancedb`, `pyarrow`, `duckdb`, `fsspec`

**Optional**:

| Extra | Packages | Purpose |
|---|---|---|
| `sprkkr` | `ase2sprkkr` | SPR-KKR file parsing (required for ingestion) |
| `ase` | `ase` | ASE Atoms interface |
| `ml` | `scikit-learn`, `pandas`, `torch` | ML feature extraction and PyTorch streaming |
| `nomad` | `nomad-lab`, `nomad-simulations` | NOMAD/FAIRmat export |
| `cloud` | `s3fs`, `gcsfs` | S3 and GCS remote storage |

---

## Development

```bash
git clone https://github.com/ridhaeddhib/oscarpes.git
cd oscarpes
pip install -e ".[dev]"
pytest
```

Build documentation:

```bash
cd docs
make html
```

---

## License

[MIT](LICENSE)
