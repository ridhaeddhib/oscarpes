"""
oscarpes
===========
OSCARpes v3 — Lance + Zarr database, parsers, visualisation,
post-processing and ML feature extraction for SPR-KKR ARPES calculations.

Quick start::

    from oscarpes import OSCARDatabase, ingest_directory
    eid = ingest_directory('/path/to/calc', formula='WSe2')   # → ~/.oscarpes/
    db  = OSCARDatabase()                                     # → ~/.oscarpes/
    e   = db[eid]

ase2sprkkr dependency
---------------------
The SPC / SPEC.out / INP parsers call into ``ase2sprkkr`` at runtime
(lazy imports inside the parser functions).  If ``ase2sprkkr`` is not
installed these parsers will raise ``ImportError``; all other
oscarpes functionality (Lance query, visualise, postprocess, ml) works
without it.
"""
from .parsers import (
    parse_inp, parse_pot, parse_spc, parse_spec_out,
    parse_arpes_out, parse_job_script,
    parse_sfn, find_files,
    sha256_file, sha256_crystal, sha256_lkkr_geom, sha256_pot,
    SpecOutData,
)
from ase2sprkkr.sprkkr.shape_function import ShapeFunction, ShapeFunctionMesh
from .ingest   import write_lance, ingest_directory, ingest_tree, compute_embeddings, DEFAULT_DB
from .entry    import (
    OSCAREntry, ARPESData,
    CrystalData, LKKRGeometryData,
    SCFData, StructureData,
    PhotonSourceData, ElectronAnalyserData, SampleData,
    # backward-compat aliases
    PhotonData, PhotoemissionData,
)
from .database import OSCARDatabase
from . import visualize
from . import postprocess
from . import ml_features
from . import nomad_export

__version__ = "3.0.0"
__author__  = "OSCARpes contributors"

__all__ = [
    # Entry / database
    "OSCAREntry", "OSCARDatabase", "ARPESData",
    "CrystalData", "LKKRGeometryData",
    "SCFData", "StructureData",
    "PhotonSourceData", "ElectronAnalyserData", "SampleData",
    "PhotonData", "PhotoemissionData",  # backward-compat aliases
    # Parsers
    "parse_inp", "parse_pot", "parse_spc", "parse_spec_out",
    "parse_arpes_out", "parse_job_script",
    "parse_sfn", "find_files",
    "sha256_file", "sha256_crystal", "sha256_lkkr_geom", "sha256_pot",
    "SpecOutData", "ShapeFunction", "ShapeFunctionMesh",
    # Ingest
    "write_lance", "ingest_directory", "ingest_tree", "compute_embeddings", "DEFAULT_DB",
    # Sub-modules
    "visualize", "postprocess", "ml_features", "nomad_export",
]
