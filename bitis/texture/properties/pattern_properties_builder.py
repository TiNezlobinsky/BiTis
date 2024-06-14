import numpy as np
import pandas as pd

from bitis.texture.analysis.objects_analysis import ObjectAnalysis
from .properties_builder import PropertiesBuilder
from .distribution_ellipse import DistributionEllipseBuilder


class PatternPropertiesBuilder(PropertiesBuilder):
    """
    A class that builds pattern properties for a given texture.

    Attributes:
        area_min (int): The threshold value for the area of objects in the texture.
        area_quantile (float): The quantile value for the area of objects in the texture.
    """

    def __init__(self, area_min=5, area_quantile=0.75):
        PropertiesBuilder.__init__(self)
        self.area_min = area_min
        self.area_quantile = area_quantile
        self.object_props = pd.DataFrame()
        self.pattern_props = pd.DataFrame()

    def build(self, texture, clear_border=False):
        """
        Builds pattern properties for the given texture.

        Args:
            texture: The binary texture to analyze.

        Returns:
            PatternProperties: The pattern properties of the texture.
        """
        object_analysis = ObjectAnalysis()
        self.object_props = object_analysis.build_props(texture, area_min=1,
                                                        connectivity=1,
                                                        clear_border=clear_border)

        density = self.calc_density(texture)
        elongation = self.calc_elongation(self.object_props)
        compactness = self.calc_compactness(texture, self.object_props)
        complexity = self.calc_complexity(self.object_props)
        anisotropy, orientation = self.calc_structural_anisotropy(self.object_props)

        self.pattern_props = pd.DataFrame()
        self.pattern_props['density'] = [density]
        self.pattern_props['elongation'] = [elongation]
        self.pattern_props['orientation'] = [orientation]
        self.pattern_props['compactness'] = [compactness]
        self.pattern_props['structural_anisotropy'] = [anisotropy]
        self.pattern_props['complexity'] = [complexity]
        return self.pattern_props

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

    def calc_solidity(self, props):
        """
        Calculates the solidity of objects in the texture.

        Args:
            props: The properties of objects in the texture.

        Returns:
            float: The solidity of objects in the texture.
        """
        quant = np.quantile(props['area'].values, self.area_quantile)
        props = props[props['area'] > quant]
        return np.mean(props['solidity'])

    def calc_density(self, tex):
        """
        Calculates the density of objects in the texture.

        Args:
            tex: The texture array.

        Returns:
            float: The density of objects in the texture.
        """
        return np.mean(tex > 0)

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

    def calc_compactness(self, tex, props):
        """
        Calculates the compactness of objects in the texture.

        Args:
            tex: The texture array.
            props: The properties of objects in the texture.

        Returns:
            float: The compactness of objects in the texture.
        """
        area_array = props['area'].values
        quant = np.quantile(area_array, self.area_quantile)
        area_array = area_array[area_array > quant]
        return area_array.sum() / tex.size
