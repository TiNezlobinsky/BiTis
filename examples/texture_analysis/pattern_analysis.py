from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, segmentation

from bitis.texture.texture import Texture
from bitis.texture.properties import (
    PatternPropertiesBuilder,
    DistributionEllipseBuilder,
    PolarPlots
)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def swap_axis(angle):
    """
    Swap axis for polar plot.
    """
    return 0.5 * np.pi - angle


def draw_anisotropy(ax, objects_props, n_std=2):
    objects_props = objects_props[objects_props['area'] >= 5]
    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse = dist_ellipse_builder.build(objects_props)
    full_theta = swap_axis(dist_ellipse.full_theta)
    orientation = swap_axis(dist_ellipse.orientation)
    r, theta, d = PolarPlots.sort_by_density(objects_props['axis_ratio'],
                                             swap_axis(objects_props['orientation']))

    ax.scatter(theta, r, c=d, s=30, alpha=1, cmap='viridis')
    ax.plot(full_theta, dist_ellipse.full_radius, color='red')

    ax.quiver(0, 0, orientation, 0.5 * dist_ellipse.width,
              angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(0, 0, 0.5 * np.pi + orientation,
              0.5 * dist_ellipse.height,
              angles='xy', scale_units='xy', scale=1, color='red')


path = Path(__file__).parents[2].joinpath('data')

for i in range(1, 10):
    im = plt.imread(path.joinpath('original_texs', f'or_tex_{i}.png'))
    nim = np.where(rgb2gray(im) > 0.5, 1, 2)

    nim_gen = np.load(path.joinpath('sim_dir_2', f'or_tex_{i}.npy'))

    nim_uni = np.random.random(nim_gen.shape) < (np.sum(nim == 2) / nim.size)

    nim_uni = nim_uni.astype(int) + 1

    textures = []
    pattern_props = []

    for im in [nim, nim_gen, nim_uni]:
        # im = ErosionSegmentation.segment(im == 2)
        pattern_builder = PatternPropertiesBuilder()
        pattern_properties = pattern_builder.build(im == 2)
        pattern_props.append(pattern_properties)

        texture = Texture()
        texture.matrix = im
        texture.properties["pattern"] = pattern_properties
        texture.properties["object_props"] = pattern_builder.object_props
        textures.append(texture)

    print(pd.concat(pattern_props))

    fig, axs = plt.subplot_mosaic([['im_or', 'im_gen', 'im_uni'],
                                   ['plot', 'plot_gen', 'plot_uni']],
                                  per_subplot_kw={('plot', 'plot_gen',
                                                   'plot_uni'): {'projection': 'polar'}})

    axs['im_gen'].sharex(axs['im_or'])
    axs['im_gen'].sharey(axs['im_or'])

    axs['im_uni'].sharex(axs['im_or'])
    axs['im_uni'].sharey(axs['im_or'])

    axs['plot'].sharex(axs['plot_gen'])
    axs['plot'].sharey(axs['plot_gen'])

    axs['plot_uni'].sharex(axs['plot_gen'])
    axs['plot_uni'].sharey(axs['plot_gen'])

    draw_anisotropy(axs['plot'], textures[0].properties["object_props"])
    draw_anisotropy(axs['plot_gen'], textures[1].properties["object_props"])
    draw_anisotropy(axs['plot_uni'], textures[2].properties["object_props"])

    axs['im_or'].imshow(textures[0].matrix, origin='lower')
    axs['im_or'].set_title('Original texture')

    axs['im_gen'].imshow(textures[1].matrix, origin='lower')
    axs['im_gen'].set_title("Generated texture")

    axs['im_uni'].imshow(textures[2].matrix, origin='lower')
    axs['im_uni'].set_title("Generated texture 2")

    plt.show()
