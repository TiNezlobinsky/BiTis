.. BiTis documentation master file

BiTis: Binarized Texture-based Image Synthesis for Cardiac Fibrosis
===================================================================

BiTis is a Python framework for generating realistic fibrosis patterns using
texture-based sampling from reference histological images.
It supports 2D data, provides patch-based synthesis tools, and includes metrics
for structural analysis.

Features
--------

- Synthetic fibrosis generation using reference images.
- Morphological and statistical analysis tools.

Installation
------------

Clone and install BiTis with pip:

.. code-block:: bash

   git clone --branch fibsim https://github.com/TiNezlobinsky/BiTis.git
   cd BiTis
   pip install -e .

Basic Usage
-----------

.. code-block:: python
    
    import bitis as bt
    from pathlib import Path
    import ast
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load the dataset
    df = bt.datasets.tissue_dataset()
    texture = df["Tissue Matrix"].iloc[1]

    # Convert the texture to a binary matrix
    training_tex = np.where(texture == 0, 1, 2).astype(np.float32)

    # Initialize the simulation
    simulation_tex = np.zeros_like(training_tex)
    simulation = bt.AdaptiveSampling(simulation_tex,
                                     training_tex,
                                     max_known_pixels=30,
                                     min_known_pixels=5,
                                     max_template_size=50,
                                     min_template_size=5,
                                     n_candidates=1)

    simulated_tex = simulation.run()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].imshow(training_tex, origin='lower')
    ax[1].imshow(simulated_tex, origin='lower')
    plt.show()
