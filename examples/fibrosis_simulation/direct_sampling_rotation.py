import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

from bitis.tissue_models.direct_sampling.precondition_builders.texture_precondition_builder import TexturePreconditionBuilder
from bitis.tissue_models.direct_sampling.simulation_path_builder.simulation_random_path_builder import SimulationRandomPathBuilder
from bitis.tissue_models.direct_sampling.training_data_builders.rotated_data_builder import RotatedDataBuilder
from bitis.tissue_models.direct_sampling.simulations.texture_simulation import TextureSimulation


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


simulation_size = np.array([100, 100])
template_size   = np.array([12, 12])

texture_precondition_builder = TexturePreconditionBuilder(simulation_size)
precondition_matrix = np.zeros(simulation_size)

simulation_path_builder = SimulationRandomPathBuilder()
simulation_path = simulation_path_builder.build(template_size, simulation_size)

original_tex = np.load("../data/example_texture.npy")
training_textures_set = [original_tex]


simulations = []
for angle in [0, 30, 60]:
    angles_matrix = np.ones([len(training_textures_set), 1]) * angle
    training_data_builder = RotatedDataBuilder(template_size, training_textures_set, angles_matrix)
    training_data, meta_data = training_data_builder.build()

    simulation = TextureSimulation()
    simulation.precondition_matrix = precondition_matrix
    simulation.simulation_path     = simulation_path
    simulation.training_data       = training_data
    simulation.training_meta       = meta_data
    simulation.labeled_matrix      = angles_matrix

    simul_tex = morphology.remove_small_objects(simulation.run() > 1, 2)
    simulations.append(simul_tex)


fig, ax = plt.subplots(1, 4)

ax[0].imshow(original_tex)
ax[0].title.set_text('Original')
ax[1].imshow(simulations[0])
ax[1].title.set_text('0 degrees')
ax[2].imshow(simulations[1])
ax[2].title.set_text('30 degrees')
ax[3].imshow(simulations[2])
ax[3].title.set_text('60 degrees')
plt.show()