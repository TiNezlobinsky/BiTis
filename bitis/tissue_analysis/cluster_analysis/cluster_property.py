import numpy as np
import pandas as pd
from skimage import measure, morphology, segmentation

from bitis.tissue_analysis.cluster_assembler import ClusterAssembler


class ClusterProperty:
    def __init__(self):
        pass

    @staticmethod
    def calc_measure_props(texture, prop_names):
        cluster_labels = ClusterAssembler.assemble(texture)
        return measure.regionprops_table(cluster_labels, intensity_image=texture, properties=prop_names)
    
    @staticmethod
    def calc_complexity(texture):
        props = ClusterProperty.calc_measure_props(texture, ['perimeter', 'area'])
        props['complexity'] = (props['perimeter'] ** 2
                               / (4 * np.pi * props['area']))
        return props
    
    @staticmethod
    def calc_axis_ratio(texture):
        props = ClusterProperty.calc_measure_props(texture, ['major_axis_length', 'minor_axis_length'])
        props['major_axis_length'] = np.where(props['major_axis_length'] >= 1,
                                              0.5 * props['major_axis_length'],
                                              0.5)
        props['minor_axis_length'] = np.where(props['minor_axis_length'] >= 1,
                                              0.5 * props['minor_axis_length'],
                                              0.5)
        props['axis_ratio'] = (props['major_axis_length']
                               / props['minor_axis_length'])
        return props
    
    @staticmethod
    def calc_density(texture):
        """
        Calculates the density of objects in the texture.

        Args:
            texture: The Texture object.

        Returns:
            float: The density of objects in the texture.
        """
        return np.mean(texture > 0)
    
    @staticmethod
    def calc_solidity(texture):
        """
        Calculates the solidity of objects in the texture.

        Args:
            props: The Texture object.

        Returns:
            float: The solidity of objects in the texture.
        """
        props = ClusterProperty.calc_measure_props(texture, ['area', 'solidity'])
        quant = np.quantile(props['area'].values, 0.75)
        props = props[props['area'] > quant]
        return np.mean(props['solidity'])
    
    @staticmethod
    def calc_compactness(texture):
        """
        Calculates the compactness of objects in the texture.

        Args:
            texture: The Texture object.

        Returns:
            float: The compactness of objects in the texture.
        """
        props = ClusterProperty.calc_measure_props(texture, ['area'])
        quant = np.quantile(props['area'].values, 0.75)
        props = props[props['area'] > quant]
        return props['area'].sum() / texture.size
    





    def calc_structural_anisotropy(self, props, n_std=2):
        props = props[props['area'] > self.area_min]
        r = props['axis_ratio'].values
        theta = props['orientation'].values

        r = np.concatenate([r, r])
        theta = np.concatenate([theta, theta + np.pi])

        dist_ellipse = DistributionEllipseBuilder().build(r, theta, n_std=n_std)
        return dist_ellipse.anisotropy, dist_ellipse.orientation

    def calc_complexity(self, props):
        """
        Calculates the complexity of objects in the texture.

        Args:
            props: The properties of objects in the texture.

        Returns:
            float: The complexity of objects in the texture.
        """
        props = props[props['area'] > self.area_min]
        return np.sum(props['complexity'])

    def calc_elongation(self, props):
        """
        Calculates the elongation of objects in the texture.

        Args:
            props: The properties of objects in the texture.

        Returns:
            float: The elongation of objects in the texture.
        """
        props = props[props['area'] > self.area_min]
        return np.median(props['axis_ratio'])
