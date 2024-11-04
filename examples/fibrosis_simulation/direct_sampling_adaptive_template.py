from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology


from bitis.tissue_models.direct_sampling.precondition_builders.texture_precondition_builder import TexturePreconditionBuilder
from bitis.tissue_models.direct_sampling.simulation_path_builder.simulation_random_path_builder import SimulationRandomPathBuilder
from bitis.tissue_models.direct_sampling.simulations.adaptive_simulation import AdaptiveSimulation
from bitis.tissue_models.direct_sampling.training_data_builders.distance_builder import DistanceBuilder



df = pd.read_csv(Path("datasets") / "tissue_dataset.csv")

# Convert string representation of numpy arrays to numpy arrays:
df['Tissue Matrix'] = df['Tissue Matrix'].apply(lambda x: np.array(ast.literal_eval(x)))
df['Tissue size'] = df['Tissue size'].apply(lambda x: ast.literal_eval(x))

# Filter the dataset to extract one specific texture that meets the criteria:
filtered_df = df[(df['Density'] >= 0.2) & (df['Density'] <= 0.25) & (df['Elongation'] > 2.5)]
texture = filtered_df["Tissue Matrix"].iloc[0]

# 1 - healthy tissue, 2 - fibrosis
texture = np.where(texture == 0, 1, 2)

# mask = morphology.remove_small_objects(texture == 2, min_size=5, connectivity=1)
# texture = np.zeros_like(texture)
# texture[mask] = 2
# texture[~mask] = 1

simulation_size = texture.shape

template_size = np.array([13, 13])

texture_precondition_builder = TexturePreconditionBuilder(simulation_size)
precondition_matrix = np.zeros(simulation_size)

simulation_path_builder = SimulationRandomPathBuilder(template_size, simulation_size)

mask = np.random.random(texture.shape) < 0.01
precondition_matrix[mask] = texture[mask]

simulation = AdaptiveSimulation(precondition_matrix, texture, simulation_path_builder, DistanceBuilder(texture))

simulated_tex = simulation.run()

# mask = morphology.remove_small_objects(simulated_tex == 2, min_size=5, connectivity=1)
# simulated_tex = np.zeros_like(simulated_tex)
# simulated_tex[mask] = 2
# simulated_tex[~mask] = 1

fig, ax = plt.subplots(1, 2)

ax[0].imshow(texture)
ax[1].imshow(simulated_tex)
plt.show()