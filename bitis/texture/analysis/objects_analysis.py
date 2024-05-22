import numpy as np
import pandas as pd

from skimage import measure, morphology, segmentation


class ObjectAnalysis:
    """
    A class that performs analysis on objects in a texture.

    Methods:
    - calc_density(tex): Calculates the density of objects in the texture.
    - build_props(tex, area_threshold, connectivity, clear_border): Builds properties of objects in the texture.
    - calc_anisotropy(props): Calculates the anisotropy of objects in the texture.
    - calc_compactness(tex, props): Calculates the compactness of objects in the texture.
    - get_pattern(tex, area_threshold, connectivity, clear_border): Gets the pattern information of objects in the texture.
    """

    def __init__(self):
        pass

    def build_props(self, tex, area_threshold=5, connectivity=1,
                    clear_border=False):
        """
        Builds properties of objects in the texture.

        Parameters:
        - tex: The texture array.
        - area_threshold: The minimum area threshold for objects to be considered.
        - connectivity: The connectivity used for labeling objects.
        - clear_border: Whether to clear objects at the border of the image.

        Returns:
        - The properties of objects in the texture.
        """
        mask = morphology.remove_small_objects(tex > 1, area_threshold)

        if clear_border:
            mask = segmentation.clear_border(mask)

        labels = measure.label(mask, background=False,
                               connectivity=connectivity)
        props = measure.regionprops_table(labels,
                                          properties=('label',
                                                      'area',
                                                      'centroid',
                                                      'solidity',
                                                      'major_axis_length',
                                                      'minor_axis_length',
                                                      'orientation',
                                                      'perimeter_crofton'))

        props['complexity'] = (props['perimeter_crofton'] ** 2
                               / (4 * np.pi * props['area']))

        props['major_axis_length'] = np.where(props['major_axis_length'] == 0,
                                              0.5,
                                              0.5 * props['major_axis_length'])
        props['minor_axis_length'] = np.where(props['minor_axis_length'] == 0,
                                              0.5,
                                              0.5 * props['minor_axis_length'])
        props['axis_ratio'] = (props['major_axis_length']
                               / props['minor_axis_length'])

        props = pd.DataFrame(props)
        return props
