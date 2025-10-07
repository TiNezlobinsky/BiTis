import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import bitis as bt


df = bt.tissue_dataset()

# Filter the dataset to extract one specific texture that meets the criteria:
filtered_df = df[(df['Density'].between(0.3, 0.35)) &
                 (df['Elongation'] > 2.5)]
texture = filtered_df["Tissue Matrix"].iloc[1]

# 1 - healthy tissue, 2 - fibrosis
texture = np.where(texture == 0, 1, 2)
training_image = texture.astype(np.float32)
angle_map = np.zeros_like(training_image)
angle_map[30:60, :] = 30
angle_map[60:, :] = 60

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
simulation.template_matching = bt.RotatedImagesMatching(training_image,
                                                        angle_map,
                                                        base_angle=60,
                                                        num_of_candidates=num_of_candidates,
                                                        min_known_pixels=min_known_pixels)
simulated_tex = simulation.run()

fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
ax[0].imshow(training_image)
ax[1].imshow(simulated_tex)
ax[2].imshow(angle_map)
plt.show()
