from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology


from bitis.tissue_models.direct_sampling.precondition_builders.texture_precondition_builder import TexturePreconditionBuilder
from bitis.tissue_models.direct_sampling.simulation_path_builder.simulation_random_path_builder import SimulationRandomPathBuilder
from bitis.tissue_models.direct_sampling.simulations.adaptive_simulation import AdaptiveSimulation
from bitis.tissue_models.direct_sampling.training_data_builders.distance_builder import DistanceBuilder


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# simulation_size = np.array([100, 100])
template_size = np.array([13, 13])

im = plt.imread(Path(__file__).parents[2].joinpath('/Users/timurnezlobinskij/FrozenScience/Project_fibrosis/FibrosisAnalysis/data_gen/original_texs',
                                                   'or_tex_2.png'))
gim = rgb2gray(im)
nim = np.where(gim > 0.5, 1, 2)

mask = morphology.remove_small_objects(nim == 2, min_size=5, connectivity=1)
nim = np.zeros_like(nim)
nim[mask] = 2
nim[~mask] = 1

simulation_size = nim.shape

texture_precondition_builder = TexturePreconditionBuilder(simulation_size)
precondition_matrix = np.zeros(simulation_size)

simulation_path_builder = SimulationRandomPathBuilder(template_size, simulation_size)

mask = np.random.random(nim.shape) < 0.01
precondition_matrix[mask] = nim[mask]

simulation = AdaptiveSimulation(precondition_matrix, nim, simulation_path_builder, DistanceBuilder(nim))

simulated_tex = simulation.run()

mask = morphology.remove_small_objects(simulated_tex == 2, min_size=5, connectivity=1)
simulated_tex = np.zeros_like(simulated_tex)
simulated_tex[mask] = 2
simulated_tex[~mask] = 1

fig, ax = plt.subplots(1, 2)

ax[0].imshow(nim)
ax[1].imshow(simulated_tex)
plt.show()