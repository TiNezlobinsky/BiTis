import numpy as np

from .cluster_property_builder import (
    DistributionEllipseBuilder,
    ClusterPropertyBuilder
)


class TextureClustersProperty:
    """
    Calculates the properties of the clusters in the texture.
    """
    def __init__(self, texture, threshold=0):
        """
        Initializes the ClusterProperty object.

        Args:
            texture: The texture to calculate the properties of.
            threshold: The threshold to use for the clusters.
        """
        self.texture = texture
        self.threshold = threshold
        self.cluster_property_builder = ClusterPropertyBuilder(texture,
                                                               threshold)

    def calc_density(self):
        """
        Calculates the density of objects in the texture.

        Returns:
            float: The density of objects in the texture.
        """
        return np.mean(self.texture > self.threshold)

    def calc_solidity(self, min_area=None, quant=None):
        """
        Calculates the solidity of objects in the texture.

        Args:
            min_area: The minimum area of the clusters to use.
            quant: The quantile to use.

        Returns:
            float: The solidity of objects in the texture.
        """
        if self.cluster_property_builder.props is None:
            self.cluster_property_builder.calc_properties(('area', 'solidity'))

        if 'solidity' not in self.cluster_property_builder.props.columns:
            self.cluster_property_builder.calc_properties(('solidity',))

        if 'area' not in self.cluster_property_builder.props.columns:
            self.cluster_property_builder.calc_properties(('area',))

        props = self.cluster_property_builder.props

        if min_area is not None:
            props = props[props['area'] > min_area]
            return np.mean(props['solidity'])

        if quant is not None:
            quant = self._calc_quantile(props['area'], quant)
            res_props = props[props['area'] > quant]
            return np.mean(res_props['solidity'])

        return np.mean(props['solidity'])

    def calc_compactness(self, min_area=None, quant=None):
        """
        Calculates the compactness of texture clusters as the ratio of the
        area of clusters with certain size to the total area of clusters.

        Args:
            min_area: The minimum area of the clusters to use.
            quant: The quantile to use.

        Returns:
            float: The compactness of objects in the texture.
        """
        if self.cluster_property_builder.props is None:
            self.cluster_property_builder.calc_properties(('area',))

        if 'area' not in self.cluster_property_builder.props.columns:
            self.cluster_property_builder.calc_properties(('area',))

        props = self.cluster_property_builder.props
        total_area = props['area'].sum()

        if min_area is not None:
            props = props[props['area'] > min_area]
            return props['area'].sum() / total_area

        if quant is not None:
            quant = self._calc_quantile(props['area'], quant)
            res_props = props[props['area'] > quant]
            return res_props['area'].sum() / total_area

        return props['area'].sum() / total_area

    def calc_structural_anisotropy(self, n_std=2, min_area=5, quant=None):
        """
        Calculates the structural anisotropy of the clusters in the texture.

        Args:
            n_std: The number of standard deviations to use.
            min_area: The minimum area of the clusters to use.
            quant: The quantile to use.

        Returns:
            float: The structural anisotropy of the clusters in the texture.
        """

        if self.cluster_property_builder.props is None:
            self.cluster_property_builder.calc_properties(('area',
                                                           'major_axis_length',
                                                           'minor_axis_length',
                                                           'orientation'))

        if 'area' not in self.cluster_property_builder.props.columns:
            self.cluster_property_builder.calc_properties(('area'))

        if 'axis_ratio' not in self.cluster_property_builder.props.columns:
            self.cluster_property_builder.calc_axis_ratio()

        if 'orientation' not in self.cluster_property_builder.props.columns:
            self.cluster_property_builder.calc_properties(('orientation',))

        props = self.cluster_property_builder.props

        if min_area is not None:
            props = props[props['area'] > min_area]

        if quant is not None:
            quant = self._calc_quantile(props['area'], quant)
            props = props[props['area'] > quant]

        r = props['axis_ratio'].values
        theta = props['orientation'].values

        r = np.concatenate([r, r])
        theta = np.concatenate([theta, theta + np.pi])

        dist_ellipse = DistributionEllipseBuilder().build(r, theta,
                                                          n_std=n_std)
        return dist_ellipse.anisotropy, dist_ellipse.orientation

    def _calc_quantile(self, prop, quant):
        """
        Calculates the quantile of the property.

        Args:
            prop: The property to calculate the quantile of.
            quant: The quantile to calculate.

        Returns:
            float: The quantile of the property.
        """
        print(len(prop.values))
        unique_values = np.unique(prop.values)
        print(len(unique_values))
        return np.quantile(unique_values, quant)
