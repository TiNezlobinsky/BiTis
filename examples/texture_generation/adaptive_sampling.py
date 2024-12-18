from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bitis as bt


df = pd.read_csv(Path("datasets") / "tissue_dataset.csv")

# Convert string representation of numpy arrays to numpy arrays:
df['Tissue Matrix'] = df['Tissue Matrix'].apply(
    lambda x: np.array(ast.literal_eval(x))
)
df['Tissue size'] = df['Tissue size'].apply(
    lambda x: ast.literal_eval(x)
)

# Filter the dataset to extract one specific texture that meets the criteria:
filtered_df = df[(df['Density'].between(0.3, 0.35)) &
                 (df['Elongation'] > 2.5)]
texture = filtered_df["Tissue Matrix"].iloc[0]

# 1 - healthy tissue, 2 - fibrosis
texture = np.where(texture == 0, 1, 2)

texture_ = texture[:, :]
simulation_tex = np.zeros(texture_.shape)
simulation = bt.AdaptiveSampling(simulation_tex,
                                 texture_,
                                 max_known_pixels=30,
                                 max_template_size=100,
                                 min_template_size=5)

simulated_tex = simulation.run()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(texture_)
ax[1].imshow(simulated_tex)
plt.show()
