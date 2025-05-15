import pandas as pd
import numpy as np
from skimage import measure, segmentation


class ClusterPropertyBuilder:
    """
    Assembles the clusters in the texture.
    """

    def __init__(self, texture, threshold=0):
        self.texture = texture
        self.threshold = threshold
        self.labeled_texture = None
        self.props = None

    def assemble(self, connectivity=1, clear_border=False):
        """
        Assembles the clusters in the texture.

        Args:
            texture: The texture to assemble the clusters in.
            connectivity: The connectivity of the clusters.
            clear_border: Whether to clear the border of the clusters.

        Returns:
            The labeled clusters.
        """
        mask = self.texture > self.threshold

        if clear_border:
            mask = segmentation.clear_border(mask)

        self.labeled_texture = measure.label(mask,
                                             background=False,
                                             connectivity=connectivity)

        return self.labeled_texture

    def measure_props(self, prop_names):
        """
        Calculates the properties of the clusters in the texture.

        Args:
            prop_names: The properties to calculate.

        Returns:
            The properties of the clusters in the texture.
        """
        if self.labeled_texture is None:
            self.assemble()

        props = measure.regionprops_table(self.labeled_texture,
                                          properties=prop_names)
        props = pd.DataFrame(props)
        return props

    def calc_properties(self, prop_names=('area', 'perimeter',
                                          'major_axis_length',
                                          'minor_axis_length',
                                          'orientation',
                                          'solidity')):
        """
        Calculates the properties of the clusters in the texture.
        """
        if self.props is None:
            self.props = self.measure_props(prop_names)
            return self.props

        props = self.measure_props(prop_names)
        for prop in prop_names:
            self.props[prop] = props[prop]

        return self.props

    def calc_axis_ratio(self):
        """
        Calculates the axis ratio of the clusters in the texture.
        """
        if self.props is None:
            self.props = self.measure_props(('major_axis_length',
                                             'minor_axis_length'))

        if 'major_axis_length' not in self.props.columns:
            props = self.measure_props(('major_axis_length',))
            self.props['major_axis_length'] = props['major_axis_length']

        if 'minor_axis_length' not in self.props.columns:
            props = self.measure_props(('minor_axis_length',))
            self.props['minor_axis_length'] = props['minor_axis_length']

        self.props['major_axis_length'] = np.where(
            self.props['major_axis_length'] >= 1,
            0.5 * self.props['major_axis_length'],
            0.5)
        self.props['minor_axis_length'] = np.where(
            self.props['minor_axis_length'] >= 1,
            0.5 * self.props['minor_axis_length'],
            0.5)
        self.props['axis_ratio'] = (self.props['major_axis_length']
                                    / self.props['minor_axis_length'])
        return self.props
