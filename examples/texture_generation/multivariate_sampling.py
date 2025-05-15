from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import bitis as bt


df = pd.read_csv(Path("/Users/arstanbek/Projects/fibrosis/BiTis/datasets") / "tissue_dataset.csv")

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

print(filtered_df.shape)
texture = filtered_df["Tissue Matrix"].iloc[0]
texture = np.where(texture == 0, 1, 2).astype(np.float32)

training_image = gaussian_filter(texture.astype(np.float32), sigma=5)

max_known_pixels = 30
max_template_size = 40
min_template_size = 3
num_of_candidates = 2
min_known_pixels = 1

simulation_tex = np.zeros_like(training_image)
simulation = bt.Simulation()
simulation.path_builder = bt.RandomSimulationPathBuilder(simulation_tex)
simulation.template_builder = bt.AdaptiveTemplateBuilder(simulation_tex,
                                                         max_known_pixels,
                                                         max_template_size,
                                                         min_template_size)
simulation.template_matching = bt.ContinuousVariableMatching(training_image,
                                                             num_of_candidates,
                                                             min_known_pixels)
simulated_tex = simulation.run()
joint_simulated_image = simulated_tex.copy()
joint_training_image = training_image.copy()

fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
fig.colorbar(ax[0].imshow(joint_training_image))
fig.colorbar(ax[1].imshow(joint_simulated_image))
fig.colorbar(ax[2].imshow(simulation._index_map))
plt.show()


training_image = texture.copy()

max_known_pixels = 30
max_template_size = 40
min_template_size = 3
n_candidates = 3
min_known_pixels = 1

simulation_tex = np.zeros_like(training_image)
multi_var_simulation = bt.Simulation()
multi_var_simulation.path_builder = bt.MultivariateSimulationPathBuilder(simulation_tex)
multi_var_simulation.template_builder = bt.AdaptiveTemplateBuilder(simulation_tex,
                                                                   max_known_pixels,
                                                                   max_template_size,
                                                                   min_template_size)
multi_var_simulation.template_matching = bt.MultivariateVariableMatching(training_image,
                                                                         joint_training_image,
                                                                         joint_simulated_image,
                                                                         n_candidates,
                                                                         min_known_pixels)

simulated_tex = multi_var_simulation.run()

fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
ax[0].imshow(training_image)
ax[1].imshow(simulated_tex)
ax[2].imshow(joint_training_image)
plt.show()
