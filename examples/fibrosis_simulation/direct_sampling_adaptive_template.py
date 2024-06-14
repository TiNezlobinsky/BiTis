from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm

from bitis.tissue_models.direct_sampling.simulations import TextureAdaptiveSimulation
from image_parser import ImageParser


def generate_image(i, j):
    simulation_image = np.zeros((200, 200))
    simulation = TextureAdaptiveSimulation(training_image,
                                           simulation_image,
                                           max_size=20,
                                           max_distance=30,
                                           min_distance=1)
    simulation.run()
    filename = f'gen_tex_{i}_{j}.png'
    ImageParser.write_png(simulation_image,
                          path.joinpath('simulated_2_1', filename))


path = Path(__file__).parents[2].joinpath('data')

for i in range(2, 3):
    filename = f'or_tex_{i}.png'
    training_image = ImageParser.read_png(path.joinpath('original_texs',
                                                        filename))
    training_image[training_image == 0] = 1
    training_image = training_image

    Parallel(n_jobs=8)(delayed(generate_image)(i, j) for j in tqdm(range(16, 32)))

# template_sizes = np.array(simulation.template_sizes)

# plt.figure()
# plt.scatter(template_sizes[:, 0], template_sizes[:, 1],
#             c=np.arange(len(template_sizes)))
# plt.show()

# fig, axs = plt.subplots(1, 2, figsize=(8, 3), width_ratios=[1, 1.5])
# axs[0].imshow(training_image, origin='lower', vmin=0, vmax=2)
# axs[0].set_title('Training Image')
# axs[0].axis('off')

# axs[1].imshow(simulation_image[25:125, 25:125], origin='lower', vmin=0, vmax=2)
# axs[1].set_title('Simulated Image')
# axs[1].axis('off')
# plt.show()
