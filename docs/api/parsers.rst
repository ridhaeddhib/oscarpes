oscarpes.parsers
===================

File parsers for every SPR-KKR output format.

.. rubric:: ase2sprkkr parsing backend

Every parser in this module delegates to the **ase2sprkkr** library:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parser function
     - File type
     - ase2sprkkr component
   * - :func:`parse_spc`
     - ``*_data.spc``
     - ``ARPESOutputFile.from_file()``
   * - :func:`parse_spec_out`
     - ``*_SPEC.out``
     - ``SpecResult``
   * - :func:`parse_inp`
     - ``*.inp``
     - ``InputParameters.from_file()``
   * - :func:`parse_pot`
     - ``*.pot`` / ``*pot_new``
     - ``Potential.from_file()``
   * - :func:`parse_sfn`
     - ``*.sfn``
     - ``read_shape_function()`` / ``ShapeFunction``

.. automodule:: oscarpes.parsers
   :members:
   :undoc-members: False
   :show-inheritance:
