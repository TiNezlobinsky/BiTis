import numpy as np
import matplotlib.pyplot as plt


from bitis.tissue_models.direct_sampling.precondition_builders.texture_precondition_builder import TexturePreconditionBuilder
from bitis.tissue_models.direct_sampling.simulation_path_builder.simulation_random_path_builder import SimulationRandomPathBuilder
from bitis.tissue_models.direct_sampling.training_data_builders.training_data_builder import TrainingDataBuilder
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

training_data_builder = TrainingDataBuilder(template_size, training_textures_set)
training_data = training_data_builder.build()

simulation = TextureSimulation()
simulation.precondition_matrix = precondition_matrix
simulation.simulation_path     = simulation_path
simulation.training_data       = training_data

simulated_tex = simulation.run()

fig, ax = plt.subplots(1,2)

ax[0].imshow(original_tex)
ax[1].imshow(simulated_tex)
plt.show()