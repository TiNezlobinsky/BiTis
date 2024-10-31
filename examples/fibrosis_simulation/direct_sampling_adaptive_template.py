from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
import joblib
import contextlib
=======
from joblib import Parallel, delayed
>>>>>>> 9b45990efab06004cfe72fbc4714b439c9dce8fd
from tqdm import tqdm

from bitis.tissue_models.direct_sampling.simulations import TextureAdaptiveSimulation
from image_parser import ImageParser


def generate_image(i, j):
<<<<<<< HEAD
    path = Path(__file__).parents[2].joinpath('data')
=======
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
>>>>>>> 9b45990efab06004cfe72fbc4714b439c9dce8fd
    filename = f'or_tex_{i}.png'
    training_image = ImageParser.read_png(path.joinpath('original_texs',
                                                        filename))
    training_image[training_image == 0] = 1
    training_image = training_image

<<<<<<< HEAD
    simulation_image = np.zeros(training_image.shape)
    simulation = TextureAdaptiveSimulation(training_image,
                                           simulation_image,
                                           max_size=20,
                                           max_distance=30,
                                           min_distance=1,
                                           distance_threshold=0.0,
                                           progress_bar=False)
    simulation.run()
    filename = f'gen_tex_{i}_{j}.png'

    if not path.joinpath(f'simulated_{i}').exists():
        path.joinpath(f'simulated_{i}').mkdir()

    ImageParser.write_png(simulation_image,
                          path.joinpath(f'simulated_{i}', filename))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


for i in [5]:
    with tqdm_joblib(tqdm(desc="Generating images", total=100)) as progress_bar:
        joblib.Parallel(n_jobs=4)(joblib.delayed(generate_image)(i, j) for j in range(100))
=======
    Parallel(n_jobs=8)(delayed(generate_image)(i, j) for j in tqdm(range(16, 32)))
>>>>>>> 9b45990efab06004cfe72fbc4714b439c9dce8fd

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
