from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.covariance import EmpiricalCovariance
from skimage import measure, morphology, segmentation

from bitis.texture.texture import Texture
from bitis.texture.properties import (
    PatternPropertiesBuilder,
    DistributionEllipseBuilder,
    PolarPlots,
    PointDensity
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
    r = objects_props['axis_ratio'].values
    theta = objects_props['orientation'].values

    r = np.concatenate([r, r])
    theta = np.concatenate([theta, theta + np.pi])

    theta = swap_axis(theta)

    dist_ellipse_builder = DistributionEllipseBuilder()
    dist_ellipse = dist_ellipse_builder.build(r, theta, n_std=n_std)
    full_theta = dist_ellipse.full_theta
    orientation = dist_ellipse.orientation
    r, theta, d = PolarPlots.sort_by_density(r, theta)

    ax.scatter(theta, r, c=d, s=30, alpha=1, cmap='viridis')
    ax.plot(full_theta, dist_ellipse.full_radius, color='red')

    ax.quiver(0, 0, orientation, 0.5 * dist_ellipse.width,
              angles='xy', scale_units='xy', scale=1, color='red')
    ax.quiver(0, 0, 0.5 * np.pi + orientation,
              0.5 * dist_ellipse.height,
              angles='xy', scale_units='xy', scale=1, color='red')

    # dist_ellipse_builder = DistributionEllipseBuilder()
    # dist_ellipse_builder.cov_estimator = EmpiricalCovariance()
    # dist_ellipse = dist_ellipse_builder.build(r, theta, n_std=n_std)
    # # full_theta = swap_axis(dist_ellipse.full_theta)
    # # orientation = swap_axis(dist_ellipse.orientation)
    # ax.plot(full_theta, dist_ellipse.full_radius, color='blue')


def calc_ccdf(df):
    count = np.bincount(df['area'].values)
    area_bins = np.arange(1 + df['area'].max())

    area_bins = area_bins[1:]
    count = count[1:]
    ccdf = np.cumsum(count[::-1])[::-1] / np.sum(count)
    return area_bins, ccdf


def calc_area_cdf(df):
    count = np.bincount(df['area'].values)
    area_bins = np.arange(1 + df['area'].max())

    area_bins = area_bins[1:]
    count = count[1:]
    area = area_bins * count

    cdf = np.cumsum(area) / np.sum(area)
    return area_bins, cdf


def draw_area_cdf(ax, objects_props, label=''):
    area_bins, cdf = calc_area_cdf(objects_props)
    ax.plot(area_bins, cdf, label=label)
    ax.set_xlabel('Size')
    ax.set_ylabel('Fibrotic Tissue')
    ax.set_xscale('log')


def draw_ccdf(ax, objects_props, label=''):
    area_bins, ccdf = calc_ccdf(objects_props)
    ax.plot(area_bins, ccdf, label=label)
    ax.set_xlabel('Size')
    ax.set_ylabel('CCDF')
    ax.set_yscale('log')
    ax.set_xscale('log')


def draw_perimeter_cdf(ax, objects_props, label=''):
    count = np.bincount(objects_props['area'].values,
                        weights=objects_props['perimeter'].values)
    area_bins = np.arange(1 + objects_props['area'].max())

    area_bins = area_bins[1:]
    count = count[1:]

    cdf = np.cumsum(count)
    
    ax.plot(area_bins, cdf, label=label)
    ax.set_xlabel('Size')
    ax.set_ylabel('Perimeter')
    ax.set_xscale('log')
    ax.set_yscale('log')


path = Path(__file__).parents[2].joinpath('data')

for i in range(0, 10, 2):
    j = 11
    im = plt.imread(path.joinpath('training', f'or_tex_{j}.png'))
    nim = np.where(rgb2gray(im) > 0.5, 1, 2)
    nim = nim[:100, :100]

    im = plt.imread(path.joinpath('simulated_100_20_30',
                                  f'gen_tex_{j}_{i}.png'))
    nim_gen = np.where(rgb2gray(im) > 0.5, 1, 2)

    nim_uni = np.random.random(nim_gen.shape) < (np.sum(nim == 2) / nim.size)

    nim_uni = nim_uni.astype(int) + 1

    textures = []
    pattern_props = []

    for im in [nim, nim_gen, nim_uni]:
        # im = ErosionSegmentation.segment(im == 2)
        pattern_builder = PatternPropertiesBuilder(area_quantile=0.95)
        pattern_properties = pattern_builder.build(im == 2)
        pattern_props.append(pattern_properties)

        texture = Texture()
        texture.matrix = im
        texture.properties["pattern"] = pattern_properties
        texture.properties["object_props"] = pattern_builder.object_props
        textures.append(texture)

    print(pd.concat(pattern_props))

    fig, axs = plt.subplot_mosaic([['im_or', 'im_gen', 'im_uni'],
                                   ['plot', 'plot_gen', 'plot_uni'],
                                   ['cmpl', 'cmpl_gen', 'cmpl_uni']],
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

    # axs['cmpl'].sharex(axs['cmpl_gen'])
    # axs['cmpl'].sharey(axs['cmpl_gen'])
    # axs['cmpl_uni'].sharex(axs['cmpl_gen'])
    # axs['cmpl_uni'].sharey(axs['cmpl_gen'])

    draw_anisotropy(axs['plot'], textures[0].properties["object_props"])
    draw_anisotropy(axs['plot_gen'], textures[1].properties["object_props"])
    draw_anisotropy(axs['plot_uni'], textures[2].properties["object_props"])

    draw_ccdf(axs['cmpl_gen'],
              textures[0].properties["object_props"],
              label='Original')
    draw_ccdf(axs['cmpl_gen'],
              textures[1].properties["object_props"],
              label='DS Generator')
    draw_ccdf(axs['cmpl_gen'],
              textures[2].properties["object_props"],
              label='Uniform Generator')

    draw_area_cdf(axs['cmpl'],
                  textures[0].properties["object_props"],
                  label='Original')
    draw_area_cdf(axs['cmpl'],
                  textures[1].properties["object_props"],
                  label='DS Generator')
    draw_area_cdf(axs['cmpl'],
                  textures[2].properties["object_props"],
                  label='Uniform Generator')

    draw_perimeter_cdf(axs['cmpl_uni'],
                       textures[0].properties["object_props"],
                       label='Original')
    draw_perimeter_cdf(axs['cmpl_uni'],
                       textures[1].properties["object_props"],
                       label='DS Generator')
    draw_perimeter_cdf(axs['cmpl_uni'],
                       textures[2].properties["object_props"],
                       label='Uniform Generator')

    axs['cmpl_uni'].set_xscale('log')
    axs['cmpl_uni'].set_yscale('log')

    axs['cmpl_gen'].legend()

    axs['im_or'].imshow(textures[0].matrix, origin='lower')
    axs['im_or'].set_title('Original texture')

    axs['im_gen'].imshow(textures[1].matrix, origin='lower')
    axs['im_gen'].set_title("Generated texture")

    axs['im_uni'].imshow(textures[2].matrix, origin='lower')
    axs['im_uni'].set_title("Uniform generator")

    plt.show()
