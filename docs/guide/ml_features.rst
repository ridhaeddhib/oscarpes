ML feature extraction
=====================

:mod:`oscarpes.ml_features` converts ARPES calculations into fixed-length
128-dimensional float64 feature vectors suitable for machine learning,
similarity search, and dimensionality reduction.

All feature extraction uses the in-memory :class:`~oscarpes.entry.OSCAREntry`
object; no file I/O or ase2sprkkr calls are made at this stage.

Feature vector layout
----------------------

The 128-element vector is divided into eight physics-motivated groups:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Indices
     - Description
   * - [0:8]
     - **Photon / polarisation** — hν, θ_inc, φ_inc, Stokes s0/s1/s2/s3, work function
   * - [8:16]
     - **SCF convergence** — E_F, irel, fullpot, Lloyd, log10(rmsavv), log10(rmsavb), iterations, V_MTZ
   * - [16:24]
     - **k-grid geometry** — k_min, k_max, NK, NE, E_min, E_max, IMV, surface site index
   * - [24:40]
     - **ARPES intensity statistics** — log10 percentiles, Fermi map moments, energy/k centroids, k asymmetry
   * - [40:56]
     - **CD-ARPES descriptors** — total integrated CD, valley positions and amplitudes, asymmetry map statistics
   * - [56:72]
     - **Spin polarisation** — P percentiles, mean abs(P), Fermi-level spin texture, k-asymmetry
   * - [72:96]
     - **EDC peak descriptors** — peak energy and intensity at 8 sampled k∥ values, bandwidth
   * - [96:112]
     - **MDC peak descriptors** — peak k∥ and intensity at 8 sampled energies
   * - [112:120]
     - **Fermi surface** — dominant Fermi k, Fermi weight, width, symmetry
   * - [120:128]
     - **Structural / radial** — alat, n_layers, nq, nt, RMT/RWS filling mean and std

All features are dimensionless or in natural units (eV, Å⁻¹, Bohr).
NaN and Inf are replaced by 0.

Single-entry features
----------------------

.. code-block:: python

   from oscarpes.ml_features import extract_features, feature_names

   vec   = extract_features(e)    # ndarray, shape (128,), dtype float64
   names = feature_names()        # list of 128 human-readable strings

   # Map names to values
   for name, val in zip(names, vec):
       print(f'{name:35s}: {val:.6g}')

Batch extraction
----------------

.. code-block:: python

   from oscarpes.ml_features import batch_extract

   entries = db.find(formula='WSe2')
   X = batch_extract(entries)   # ndarray, shape (N, 128)

Pandas DataFrame
----------------

.. code-block:: python

   from oscarpes.ml_features import feature_dataframe

   df = feature_dataframe(entries)
   # DataFrame with columns: entry_id, formula, + 128 feature columns

Storing embeddings in the database
------------------------------------

Pre-computed embeddings are stored in the ``embedding`` column of the Lance
dataset for fast vector search:

.. code-block:: python

   from oscarpes.ingest import compute_embeddings

   # Compute and persist embeddings for all entries
   compute_embeddings('/data/oscar_db/')

   # Query by vector similarity (using lancedb)
   import lancedb
   db_lance = lancedb.connect('/data/oscar_db/')
   tbl = db_lance.open_table('entries')
   query_vec = extract_features(e).tolist()
   results = tbl.search(query_vec).limit(10).to_arrow()

Using with scikit-learn
------------------------

.. code-block:: python

   from sklearn.decomposition import PCA
   from sklearn.cluster import KMeans
   from oscarpes.ml_features import batch_extract

   entries = db.list_entries()
   # Load entry objects
   loaded  = [db[r['entry_id']] for r in entries]
   X       = batch_extract(loaded)

   # PCA
   pca  = PCA(n_components=2)
   X2   = pca.fit_transform(X)

   # K-means clustering
   km   = KMeans(n_clusters=4)
   labels = km.fit_predict(X)

Using with PyTorch
-------------------

.. code-block:: python

   import torch
   from oscarpes.ml_features import batch_extract

   X     = batch_extract(entries)
   X_t   = torch.tensor(X, dtype=torch.float32)

   # Or stream directly from Lance via PyTorch DataLoader
   ds     = db.as_pytorch_dataset(filter="irel = 3")
   loader = torch.utils.data.DataLoader(ds, batch_size=32)
