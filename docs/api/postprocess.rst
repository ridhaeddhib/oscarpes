oscarpes.postprocess
=======================

Post-processing routines for OSCAREntry objects.

All functions operate on in-memory :class:`~oscarpes.entry.OSCAREntry`
objects and return plain NumPy arrays or Python dicts.  No file I/O, no
matplotlib.

Peak detection uses ``scipy.signal.find_peaks``.  Integration uses
``numpy.trapezoid``.

.. automodule:: oscarpes.postprocess
   :members:
   :undoc-members: False
   :show-inheritance:
