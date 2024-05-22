from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, segmentation

from bitis.texture.texture import Texture
from bitis.texture.properties_builders.pattern_properties_builder import PatternPropertiesBuilder


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class ErosionSegmentation:
    def __init__(self) -> None:
        pass

    @staticmethod
    def segment(image, min_distance=1, min_seed_size=10):
        distance = ndimage.distance_transform_edt(image)

        seeds_mask = distance > min_distance
        seeds_mask = morphology.remove_small_objects(seeds_mask, min_seed_size)
        seeds_label, seeds_num = measure.label(seeds_mask, connectivity=1,
                                               return_num=True)
        seeds_index = np.arange(1, seeds_num + 1)

        centroids = ndimage.minimum_position(-distance, seeds_label,
                                             index=seeds_index)
        centroids = np.array(centroids)

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(centroids.T)] = True
        markers = seeds_label.copy()
        markers[~mask] = 0

        segmented = segmentation.watershed(-distance, markers,
                                           mask=image,
                                           watershed_line=True)

        non_segmented = image.copy()
        non_segmented[segmented > 0] = 0
        non_segmented_label, n = measure.label(non_segmented, connectivity=1,
                                               return_num=True)

        non_segmented_label[segmented > 0] = n + segmented[segmented > 0]

        return non_segmented_label


path = Path(__file__).parents[2].joinpath('data')

for i in range(1, 10):
    im = plt.imread(path.joinpath('original_texs', f'or_tex_{i}.png'))
    nim = np.where(rgb2gray(im) > 0.5, 1, 2)

    nim_gen = np.load(path.joinpath('sim_dir_2', f'or_tex_{i}.npy'))

    nim_gen_2 = np.random.random(nim_gen.shape) < (np.sum(nim == 2) / nim.size)

    nim_gen_2 = nim_gen_2.astype(int) + 1

    textures = []

    for im in [nim, nim_gen, nim_gen_2]:
        im = ErosionSegmentation.segment(im == 2)
        pattern_builder = PatternPropertiesBuilder()
        pattern_properties = pattern_builder.build(im)
        texture = Texture()
        texture.matrix = im
        texture.properties["pattern"] = pattern_properties
        texture.properties["object_props"] = pattern_builder.object_props

        print(texture.properties['pattern'])

        textures.append(texture)

    fig, axs = plt.subplot_mosaic([['nim', 'nim_gen', 'nim_gen_2'],
                                   ['sol_com', 'sol_com_gen', 'sol_com_gen_2']])

    axs['nim_gen'].sharex(axs['nim'])
    axs['nim_gen'].sharey(axs['nim'])

    axs['nim_gen_2'].sharex(axs['nim'])
    axs['nim_gen_2'].sharey(axs['nim'])

    axs['sol_com'].sharex(axs['sol_com_gen'])
    axs['sol_com'].sharey(axs['sol_com_gen'])

    axs['sol_com_gen_2'].sharex(axs['sol_com_gen'])
    axs['sol_com_gen_2'].sharey(axs['sol_com_gen'])

    axs['sol_com'].scatter(textures[0].properties["object_props"]['solidity'],
                           textures[0].properties["object_props"]['complexity'])

    axs['sol_com'].set_ylabel('Complexity')
    axs['sol_com'].set_xlabel('Solidity')
    axs['sol_com'].set_xlim(0, 1)

    axs['sol_com_gen'].scatter(textures[1].properties["object_props"]['solidity'],
                               textures[1].properties["object_props"]['complexity'])
    axs['sol_com_gen'].set_ylabel('Complexity')
    axs['sol_com_gen'].set_xlabel('Solidity')
    axs['sol_com_gen'].set_xlim(0, 1)

    axs['sol_com_gen_2'].scatter(textures[2].properties["object_props"]['solidity'],
                                    textures[2].properties["object_props"]['complexity'])
    axs['sol_com_gen_2'].set_ylabel('Complexity')
    axs['sol_com_gen_2'].set_xlabel('Solidity')
    axs['sol_com_gen_2'].set_xlim(0, 1)

    axs['nim'].imshow(textures[0].matrix)
    axs['nim'].set_title('Original texture')

    axs['nim_gen'].imshow(textures[1].matrix)
    axs['nim_gen'].set_title("Generated texture")

    axs['nim_gen_2'].imshow(textures[2].matrix)
    axs['nim_gen_2'].set_title("Generated texture 2")

    plt.show()
