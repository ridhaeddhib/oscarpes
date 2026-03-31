"""
oscarpes.nomad_plugin
========================
NOMAD parser plugin for SPR-KKR ARPES calculations.

This is the first KKR photoemission parser in the NOMAD/FAIRmat ecosystem.
It wraps oscarpes.parsers (no additional parsing logic needed) and
populates a custom NOMAD metainfo schema that aligns with NXmpes_arpes vocabulary.

Plugin registration
-------------------
In ``pyproject.toml`` add::

    [project.entry-points."nomad.plugin"]
    sprkkr_parser = "oscarpes.nomad_plugin:sprkkr_parser"
    sprkkr_schema = "oscarpes.nomad_plugin:sprkkr_schema"

Schema hierarchy
----------------
::

    run
    ├── ModelSystem (AtomicCell + AtomsState)   — reuses nomad-simulations
    ├── ModelMethod
    │   └── KKRMethod                            — CUSTOM extends DFT
    │       └── RadialPotential[]
    └── TheoreticalARPES                         — CUSTOM output
        ├── PhotonSource
        ├── DetectorGeometry
        ├── SemiInfiniteSurface  (miller_hkl, iq_at_surf, work_function_ev)
        └── ARPESOutput
"""
try:
    from nomad.config.models.plugins import ParserEntryPoint, SchemaPackageEntryPoint

    sprkkr_parser = ParserEntryPoint(
        name='SPRKKRParser',
        description=(
            'NOMAD parser for SPR-KKR ARPES calculations (oscarpes). '
            'First KKR photoemission parser in the FAIRmat ecosystem.'
        ),
        mainfile_name_re=r'.*\.inp$',
        mainfile_mime_re=r'text/.*',
        python_package='oscarpes.nomad_plugin.parser',
    )

    sprkkr_schema = SchemaPackageEntryPoint(
        name='SPRKKRSchema',
        description='NOMAD metainfo schema for SPR-KKR ARPES (oscarpes v3)',
        python_package='oscarpes.nomad_plugin.schema',
    )

except ImportError:
    # nomad-lab not installed — plugin registration skipped silently
    sprkkr_parser = None
    sprkkr_schema = None
