import numpy as np
import matplotlib.pyplot as plt


from bitis.tissue_models.direct_sampling.precondition_builder.texture_precondition_builder import TexturePreconditionBuilder
from bitis.tissue_models.direct_sampling.simulation_path_builder.simulation_random_path_builder import SimulationRandomPathBuilder
from bitis.tissue_models.direct_sampling.training_data_builder.training_data_builder import TrainingDataBuilder
from bitis.tissue_models.direct_sampling.simulation.texture_simulation import TextureSimulation


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


simulation_size = np.array([100, 100])
template_size   = np.array([9, 9])

texture_precondition_builder = TexturePreconditionBuilder(simulation_size)
precondition_matrix = np.zeros(simulation_size)

simulation_path_builder = SimulationRandomPathBuilder()
simulation_path = simulation_path_builder.build(template_size, simulation_size)

im = plt.imread("../../../MPS_generator/original_texs/or_tex_34.png")
gim = rgb2gray(im)
nim = np.where(gim > 0.5, 1, 2)
training_textures_set = [nim]

training_data_builder = TrainingDataBuilder(template_size, training_textures_set)
training_data = training_data_builder.build()

simulation = TextureSimulation()
simulation.precondition_matrix = precondition_matrix
simulation.labeled_matrix      = None
simulation.simulation_path     = simulation_path
simulation.training_data       = training_data
simulation.trainig_meta        = None
simulation.simulation_kernel   = None

simulated_tex = simulation.run()

plt.imshow(simulated_tex)
plt.show()