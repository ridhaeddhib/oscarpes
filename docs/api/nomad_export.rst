oscarpes.nomad\_export
=========================

NOMAD / FAIRmat archive export for oscarpes database entries.

Generates NOMAD-compatible JSON archives following the ``nomad-simulations``
metainfo schema.  Requires the ``[nomad]`` optional extra:

.. code-block:: bash

   pip install "oscarpes[nomad]"

The NOMAD plugin entry points (``sprkkr_parser``, ``sprkkr_schema``) are
declared in ``pyproject.toml`` and auto-discovered when ``nomad-lab`` is
installed.

.. automodule:: oscarpes.nomad_export
   :members:
   :undoc-members: False
   :show-inheritance:
