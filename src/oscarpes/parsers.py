"""
oscarpes.parsers
========================
Parsers for SPR-KKR file types.

`ase2sprkkr` remains the primary parser for SPC, SPEC.out, INP, POT, SFN,
and the main KKRSPEC `.out` file metadata. Regex is only used as a fallback
for fields that `ase2sprkkr` does not currently expose, and for HPC batch
scripts where there is no upstream parser yet.
"""

from __future__ import annotations
import os
import hashlib
import warnings
from pathlib import Path

import numpy as np



# ══════════════════════════════════════════════════════════════════════════════
#  SPC  —  use ase2sprkkr's native grammar 
# ══════════════════════════════════════════════════════════════════════════════

def parse_spc(path: str):
    """
    Parse a ``*_data.spc`` ARPES cross-section file.

    Uses the ase2sprkkr native ``ARPESOutputFile`` parser.

    Returns
    -------
    ARPESOutputFile
        With attributes: TOTAL, UP, DOWN, POLARIZATION, K, THETA, ENERGY,
        DETERMINANT (all shaped (NK, NE) in energy mode).
    """
    from ase2sprkkr.output_files.output_files import OutputFile
    return OutputFile.from_file(path, first_try='spc')


# ══════════════════════════════════════════════════════════════════════════════
#  SPEC.out  —  thin wrapper around ase2sprkkr's SpecResult
# ══════════════════════════════════════════════════════════════════════════════

def parse_spec_out(path: str):
    """
    Parse a ``*_SPEC.out`` file.

    Returns an ``ase2sprkkr.outputs.readers.spec.SpecResult`` instance loaded
    from ``path``.  All physics is parsed inside SpecResult (Stokes vector,
    photon wavevector, Jones vector, potential barrier, lattice, spectral
    data).

    Convenience attribute map (for callers that used the old SpecOutData API):

    =====================  ===================================
    old SpecOutData field  SpecResult property
    =====================  ===================================
    stokes_s0              ``.stokes['s0']``
    stokes_s1_pct          ``.stokes['s1_pct']``
    stokes_s2_pct          ``.stokes['s2_pct']``
    stokes_s3_pct          ``.stokes['s3_pct']``
    polarization_type      ``.polarization_type``
    photon_wavevector      ``.photon_wavevector``
    vector_potential_re    ``.vector_potential['re']``
    vector_potential_im    ``.vector_potential['im']``
    barrier_ibar           ``.potential_barrier['ibar']``
    barrier_epsx           ``.potential_barrier['epsx']``
    barrier_zparup         ``.potential_barrier['zparup']``
    barrier_zpardn         ``.potential_barrier['zpardn']``
    barrier_bparp          ``.potential_barrier['bparp']``
    alat_bohr              ``.lattice_constants['a']``
    basis_real             ``.basis_vectors['real']``
    basis_recip            ``.basis_vectors['reciprocal']``
    bulkrepeat             ``.bulkrepeat``
    spectral_data          ``.spectral_data``
    =====================  ===================================
    """
    from ase2sprkkr.outputs.readers.spec import SpecResult

    class _FileSpecResult(SpecResult):
        """SpecResult loaded directly from a file path (no KkrProcess context)."""
        def __init__(self, path: str):
            with open(path) as f:
                self._file_content = f.read()
            self.output_file = None

    return _FileSpecResult(path)


# ── SpecOutData: empty-content SpecResult used when no SPEC.out file exists ───

def _make_spec_result_class():
    from ase2sprkkr.outputs.readers.spec import SpecResult
    class SpecOutData(SpecResult):
        """Returned by ingest when no SPEC.out file is present.

        Subclasses SpecResult with empty content so all properties return
        their natural zero/None defaults without any file I/O.
        """
        def __init__(self):
            self._file_content = ''
            self.output_file = None
    return SpecOutData

SpecOutData = _make_spec_result_class()


# ══════════════════════════════════════════════════════════════════════════════
#  INP  —  ase2sprkkr grammar first, regex fallback for old xband format
# ══════════════════════════════════════════════════════════════════════════════

def _parse_inp_regex(path: str) -> dict:
    """Regex-based fallback parser for SPR-KKR .inp files.

    Handles files where the ase2sprkkr grammar fails (e.g. ARPES task type
    not yet recognised by the grammar).  Extracts the most important
    parameters needed for ingestion.
    """
    import re
    with open(path) as f:
        text = f.read()

    result = {}

    def _val(s):
        """Convert a raw string token to int / float / str."""
        s = s.strip().rstrip(',')
        try: return int(s)
        except ValueError: pass
        try: return float(s)
        except ValueError: pass
        return s

    def _vec(s):
        """Parse {a,b,...} or plain tokens into a list of numbers."""
        s = s.strip()
        inner = re.sub(r'[{}]', '', s)
        parts = [p.strip() for p in re.split(r'[,\s]+', inner) if p.strip()]
        vals = []
        for p in parts:
            try: vals.append(int(p))
            except ValueError:
                try: vals.append(float(p))
                except ValueError: pass
        return vals if len(vals) > 1 else (vals[0] if vals else None)

    # Strip comments and continuation lines into a flat key=value stream
    # Section headers (all-caps words at line start) set current section
    section = ''
    for line in text.splitlines():
        line = re.sub(r'#.*', '', line).strip()
        if not line:
            continue
        # Section header: all-caps word alone or followed by key=value
        sec_m = re.match(r'^([A-Z_]{2,})\s*(.*)', line)
        if sec_m:
            section = sec_m.group(1)
            rest = sec_m.group(2)
        else:
            rest = line

        # Extract all KEY=VALUE pairs from the rest of the line
        for m in re.finditer(r'([A-Z][A-Z0-9_]*)\s*=\s*(\{[^}]*\}|[^\s,]+)', rest):
            k, v = m.group(1), m.group(2)
            flat_key = k.lower()
            if v.startswith('{'):
                result[flat_key] = _vec(v)
            else:
                result[flat_key] = _val(v)

        # Standalone flags (e.g. TRANSP_BAR, FEGFINAL)
        for flag in re.findall(r'\b([A-Z][A-Z0-9_]{2,})\b', rest):
            if flag not in ('ARPES', 'XAS', section) and f'={flag}' not in rest:
                result.setdefault(flag.lower(), True)

    # Rename section-qualified collisions to match grammar-parser output
    _ALIAS = {'theta': 'theta_ph', 'phi': 'phi_ph'}
    for old, new in _ALIAS.items():
        if old in result and new not in result:
            result[new] = result.pop(old)

    result.setdefault('final_state_model',
                      'FEGFINAL' if result.pop('fegfinal', False) else 'FP')
    result.setdefault('lloyd',    False)
    result.setdefault('rel_mode', '')
    result.setdefault('mdir',     None)

    # Store raw file path for provenance
    result['_inp_path'] = path
    return result


def parse_inp(path: str) -> dict:
    """
    Parse a SPR-KKR ``*.inp`` ARPES task file via the ase2sprkkr grammar.

    Returns a flat dict with lowercase keys, normalised to plain Python
    scalars and lists (numpy arrays unwrapped).  The key names follow the
    SPR-KKR parameter names, with section-disambiguating prefixes where the
    same name appear in multiple sections (``theta_ph`` / ``theta_el``,
    ``phi_ph`` / ``phi_el``).

    Falls back to a regex parser when the ase2sprkkr grammar fails
    (e.g. ARPES task type not yet in the grammar).
    """
    try:
        from ase2sprkkr.input_parameters.input_parameters import InputParameters
        ip = InputParameters.from_file(path)
        raw = ip.as_dict()
    except Exception:
        return _parse_inp_regex(path)

    def norm(v):
        """Unwrap numpy scalars/arrays to plain Python types."""
        if isinstance(v, np.ndarray):
            return v.flat[0].item() if v.size == 1 else v.tolist()
        # unyt quantities (e.g. ImE in Ry) — just the numeric value
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
        return v

    def sec(name):
        return raw.get(name, {})

    ph = sec('SPEC_PH')
    el = sec('SPEC_EL')

    result = {}
    # flatten all sections with lowercase keys; handle name collisions explicitly
    # K and NK are numbered/repeated → handled separately below; skip here
    _SKIP   = {'SPEC_EL.K', 'SPEC_EL.NK'}
    _RENAME = {'SPEC_PH.THETA': 'theta_ph', 'SPEC_PH.PHI': 'phi_ph',
               'SPEC_EL.THETA': 'theta_el', 'SPEC_EL.PHI': 'phi_el',
               'SPEC_EL.NT':    'nt',        'SPEC_EL.NP':  'np_el'}
    for section_name, section_dict in raw.items():
        if not isinstance(section_dict, dict):
            continue
        for key, val in section_dict.items():
            qualified = f'{section_name}.{key}'
            if qualified in _SKIP:
                continue
            flat_key  = _RENAME.get(qualified, key.lower())
            result[flat_key] = norm(val)

    # K is repeated/numbered (K1={…}, K2={…}, …) → stored as object array
    # NK is repeated/numbered (NK1=N, NK2=M, …) → stored as int array
    raw_k  = sec('SPEC_EL').get('K')
    raw_nk = sec('SPEC_EL').get('NK')
    if raw_k is not None:
        try:
            for i, kv in enumerate(np.asarray(raw_k, dtype=object).flat, 1):
                result[f'k{i}'] = np.asarray(kv, dtype=float).tolist()
        except Exception:
            pass
    if raw_nk is not None:
        try:
            for i, nkv in enumerate(np.asarray(raw_nk).ravel(), 1):
                result[f'nk{i}'] = int(nkv)
        except Exception:
            pass

    # derived / special cases not directly in as_dict()
    result['final_state_model'] = 'FEGFINAL' if result.pop('fegfinal', False) else 'FP'
    result.setdefault('lloyd',    False)
    result.setdefault('rel_mode', '')
    result.setdefault('mdir',     None)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  POT  —  ase2sprkkr Potential (grammar engine)
# ══════════════════════════════════════════════════════════════════════════════

def parse_pot(path: str):
    """Parse a SPR-KKR potential file via the ase2sprkkr grammar engine.

    Returns an ``ase2sprkkr.potentials.potentials.Potential`` object.
    Access data via sections, e.g.::

        pot['SCF-INFO']['EF']()          # Fermi energy (Ry)
        pot['LATTICE']['ALAT']()         # lattice constant (Bohr)
        pot['MESH INFORMATION']['DATA']()  # radial mesh table
        pot.atoms                         # ASE Atoms object
    """
    from ase2sprkkr.potentials.potentials import Potential
    return Potential.from_file(path)






# ══════════════════════════════════════════════════════════════════════════════
#  SFN  —  shape function parser (implemented in ase2sprkkr)
# ══════════════════════════════════════════════════════════════════════════════
# SfnMeshRecord and SfnData have been replaced by ShapeFunctionMesh and
# ShapeFunction from ase2sprkkr.sprkkr.shape_function.
# Use read_shape_function(path, alat) directly, or via parse_sfn below.

from ase2sprkkr.sprkkr.shape_function import (     # noqa: E402
    ShapeFunction  as SfnData,
    ShapeFunctionMesh as SfnMeshRecord,
    read_shape_function,
)


def parse_sfn(path: str, alat: float = 1.0) -> SfnData:
    """Parse a SPR-KKR shape function file via the ase2sprkkr reader."""
    return read_shape_function(path, alat)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def sha256_file(path: str) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_crystal(pot) -> str:
    """Return a stable SHA-256 key identifying the bulk crystal.

    Accepts an ase2sprkkr ``Potential`` object.  The key is derived
    solely from lattice geometry and atom types — fields independent of
    SCF convergence.  Two potentials whose crystals hash to the same key
    share the same entry in ``crystals/``.
    """
    lat  = pot['LATTICE']
    brav = lat['BRAVAIS']() or ('', '', '', '', '')
    alat = float(lat['ALAT']() or 0.)
    cell = lat['SCALED_PRIMITIVE_CELL']()
    vecs = list(cell.ravel()) if cell is not None else [0.] * 9

    h = hashlib.sha256()
    h.update(str(brav[1]).encode())   # bravais_type
    h.update(str(brav[4]).encode())   # point_group
    for val in [alat] + vecs:
        h.update(f'{val:.10g}'.encode())
    # real atom types (Z > 0), sorted for stability
    typ_data = pot['TYPES']['DATA']()
    if typ_data is not None:
        real_types = sorted(
            [(int(row[1]), str(row[0])) for row in typ_data if int(row[1]) > 0]
        )
        for Z, label in real_types:
            h.update(f'{Z}:{label}'.encode())
    return h.hexdigest()


def sha256_lkkr_geom(path: str) -> str:
    """Return SHA-256 of the in_structur.inp file (LKKR geometry key)."""
    return sha256_file(path)


def sha256_pot(path: str) -> str:
    """Return SHA-256 of the potential file (potential key)."""
    return sha256_file(path)


def find_files(directory: str) -> dict:
    """
    Scan a calculation directory and return a dict of recognised file paths.

    Keys: 'inp', 'pot', 'spc', 'spec_out', 'sfn',
          'in_structur', 'calc_out', 'slurm_sh', 'dos', 'bsf'
    """
    d = Path(directory)
    found: dict = {}

    # .inp — prefer the ARPES task file (contains ADSI/TASK/SPEC_PH keywords)
    inp_candidates = [
        f for f in d.glob('*.inp')
        if 'structur' not in f.name.lower()
    ]
    for f in inp_candidates:
        try:
            txt = f.read_text(errors='replace')
            if 'ADSI=ARPES' in txt or 'ADSI = ARPES' in txt or 'SPEC_PH' in txt:
                found['inp'] = str(f)
                break
        except IOError:
            pass
    if 'inp' not in found and inp_candidates:
        found['inp'] = str(inp_candidates[0])

    # .pot / *pot_new
    for pat in ('*pot_new', '*.pot'):
        hits = list(d.glob(pat))
        if hits:
            found['pot'] = str(hits[0])
            break

    # *_data.spc — pick the largest; warn on a tie
    hits = list(d.glob('*_data.spc')) or list(d.glob('*.spc'))
    if hits:
        hits_sorted = sorted(hits, key=lambda p: p.stat().st_size, reverse=True)
        chosen = hits_sorted[0]
        if len(hits_sorted) > 1 and hits_sorted[0].stat().st_size == hits_sorted[1].stat().st_size:
            tied = [p.name for p in hits_sorted]
            warnings.warn(
                f"[find_files] Multiple .spc files with identical size "
                f"({chosen.stat().st_size} bytes) in {d}:\n"
                f"  {tied}\n"
                f"  Picking '{chosen.name}' (first alphabetically). "
                f"Consider removing the duplicate or renaming.",
                UserWarning, stacklevel=2,
            )
        found['spc'] = str(chosen)

    # *_SPEC.out
    hits = list(d.glob('*_SPEC.out')) or list(d.glob('*SPEC*.out'))
    if hits:
        found['spec_out'] = str(hits[0])

    # *.sfn
    hits = list(d.glob('*.sfn'))
    if hits:
        found['sfn'] = str(hits[0])

    # in_structur.inp
    hits = list(d.glob('in_structur*.inp')) or list(d.glob('in_struct*.inp'))
    if hits:
        found['in_structur'] = str(hits[0])

    # *.out (main calc log — not SPEC.out)
    for f in d.glob('*.out'):
        if 'SPEC' not in f.name:
            found['calc_out'] = str(f)
            break

    # kkrspec.sh / kkrscf.sh / SLURM script
    hits = (list(d.glob('kkrspec.sh')) or list(d.glob('kkrscf.sh'))
            or list(d.glob('slurm*.sh')) or list(d.glob('*.sh')))
    if hits:
        found['slurm_sh'] = str(hits[0])

    # DOS / BSF output files
    hits = list(d.glob('*.dos'))
    if hits:
        found['dos'] = str(hits[0])
    hits = list(d.glob('*.bsf')) or list(d.glob('*BSF*'))
    if hits:
        found['bsf'] = str(hits[0])

    return found


# ══════════════════════════════════════════════════════════════════════════════
#  ARPES .out  —  ase2sprkkr TaskResult + regex for unhandled fields
# ══════════════════════════════════════════════════════════════════════════════

def parse_arpes_out(path: str) -> dict:
    """Parse the main KKRSPEC calculation output file (``*.out``).

    Uses ``ase2sprkkr.outputs.task_result.TaskResult.from_file()`` for
    program metadata (version, executable, start timestamp).  CPU/wall
    time, compile info, MPI count, and stop status are extracted via
    regex because ase2sprkkr does not yet parse them.

    Returns
    -------
    dict
        Flat dict with keys: ``kkrspec_version``, ``kkrspec_copyright``,
        ``kkrspec_executable``, ``execution_datetime`` (ISO-8601),
        ``compiled_on``, ``compiled_with``, ``compile_date``,
        ``mpi_nprocs``, ``calc_input_file``, ``cpu_time_sec``,
        ``wall_time_sec``, ``stop_status``.  Returns ``{}`` on failure.
    """
    import re
    import datetime as _dt
    result: dict = {}

    # ── ase2sprkkr TaskResult ─────────────────────────────────────────────────
    try:
        from ase2sprkkr.outputs.task_result import TaskResult
        tr = TaskResult.from_file(path)
        pi = tr.program_info or {}
        ver = pi.get('version')
        exe = pi.get('executable')
        st  = pi.get('start_time')
        if ver is not None:
            result['kkrspec_version']    = str(ver).strip()
        if exe is not None:
            result['kkrspec_executable'] = str(exe).strip()
        if isinstance(st, _dt.datetime):
            result['execution_datetime'] = st.isoformat()
    except Exception:
        pass

    # ── regex for fields not covered by ase2sprkkr ────────────────────────────
    try:
        with open(path, errors='replace') as f:
            lines = f.readlines()
        head = ''.join(lines[:40])
        tail = ''.join(lines[-60:])

        # version + copyright from same header line
        m = re.search(
            r'KKRSPEC\s+VERSION\s+([\d.]+)\s+(\(C\)\s+\d{4}\s+[^*\n]+)',
            head,
            re.IGNORECASE,
        )
        if m:
            result.setdefault('kkrspec_version',   m.group(1).strip())
            result.setdefault('kkrspec_copyright', m.group(2).strip())

        # execution timestamp fallback (if ase2sprkkr didn't provide it)
        if 'execution_datetime' not in result:
            m = re.search(
                r'programm execution\s+on\s+(\d{2}/\d{2}/\d{4})\s+at\s+(\d{2}:\d{2}:\d{2})',
                head)
            if m:
                try:
                    dt = _dt.datetime.strptime(
                        f'{m.group(1)} {m.group(2)}', '%d/%m/%Y %H:%M:%S')
                    result['execution_datetime'] = dt.isoformat()
                except ValueError:
                    pass

        m = re.search(r'Compiled on\s*:\s*(.+)',    head)
        if m:
            result['compiled_on'] = m.group(1).strip().rstrip('*').strip()
        m = re.search(r'Compiled with\s*:\s*(.+)',  head)
        if m:
            result['compiled_with'] = m.group(1).strip().rstrip('*').strip()
        m = re.search(r'Date of compile\s*:\s*(.+)', head)
        if m:
            result['compile_date'] = m.group(1).strip().rstrip('"/').strip()

        m = re.search(r'MPI calculation with NPROCS\s*=\s*(\d+)', head)
        result['mpi_nprocs'] = int(m.group(1)) if m else 1

        m = re.search(r'input file\s*:\s*(\S+)', head)
        if m: result['calc_input_file'] = m.group(1).strip()

        m = re.search(r'run time info\s+CPU\s+([\d.]+)', tail)
        if m: result['cpu_time_sec'] = float(m.group(1))
        m = re.search(r'WALL\s+([\d.]+)', tail)
        if m: result['wall_time_sec'] = float(m.group(1))

        m = re.search(r'program stopped.*?via\s+(\w+)', tail, re.IGNORECASE)
        if m: result['stop_status'] = m.group(1).strip()

    except Exception:
        pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SLURM job script  —  regex (ase2sprkkr has no HPC job submission handling)
# ══════════════════════════════════════════════════════════════════════════════

def parse_job_script(path: str) -> dict:
    """Parse a SLURM batch job script (``*.sh``).

    Extracts ``#SBATCH`` directives and key script commands.
    Returns ``{}`` if the file does not exist or is not a SLURM script.

    Returns
    -------
    dict
        Keys: ``slurm_job_name``, ``slurm_partition``, ``slurm_ntasks``,
        ``slurm_cpus_per_task``, ``slurm_nodes``, ``slurm_mem``,
        ``slurm_time``, ``slurm_conda_env``, ``slurm_python_driver``.
    """
    import re
    result: dict = {}
    try:
        with open(path, errors='replace') as f:
            text = f.read()
        if '#SBATCH' not in text:
            return {}

        def _sbatch(flag, cast=str):
            m = re.search(rf'#SBATCH\s+--{re.escape(flag)}[=\s]+(\S+)', text)
            if not m:
                return None
            try:
                return cast(m.group(1))
            except (ValueError, TypeError):
                return m.group(1)

        result['slurm_job_name']      = _sbatch('job-name')
        result['slurm_partition']     = _sbatch('partition')
        result['slurm_ntasks']        = _sbatch('ntasks',        int)
        result['slurm_cpus_per_task'] = _sbatch('cpus-per-task', int)
        result['slurm_nodes']         = _sbatch('nodes',         int)
        result['slurm_mem']           = _sbatch('mem')
        result['slurm_time']          = _sbatch('time')

        m = re.search(r'conda\s+activate\s+(\S+)', text)
        result['slurm_conda_env']     = m.group(1) if m else None

        m = re.search(r'python\s+(\S+\.py)', text)
        result['slurm_python_driver'] = m.group(1) if m else None

    except Exception:
        pass
    return result
