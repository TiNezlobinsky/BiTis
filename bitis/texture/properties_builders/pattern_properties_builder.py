import numpy as np
import pandas as pd

from bitis.texture.properties.pattern_properties import PatternProperties
from bitis.texture.analysis.objects_analysis import ObjectAnalysis
from .properties_builder import PropertiesBuilder
from .distribution_ellipse import DistributionEllipseBuilder


class PatternPropertiesBuilder(PropertiesBuilder):
    """
    A class that builds pattern properties for a given texture.

    Attributes:
        area_threshold (int): The threshold value for the area of objects in the texture.
    """

    def __init__(self):
        PropertiesBuilder.__init__(self)
        self.area_threshold = 5
        self.object_props = pd.DataFrame()
        self.pattern_props = pd.DataFrame()

    def build(self, texture):
        """
        Builds pattern properties for the given texture.

        Args:
            texture: The texture to analyze.

        Returns:
            PatternProperties: The pattern properties of the texture.
        """
        object_analysis = ObjectAnalysis()
        props = object_analysis.build_props(texture, self.area_threshold,
                                            connectivity=1, clear_border=False)

        self.object_props = pd.DataFrame(props)

        density = self.calc_density(texture)
        elongation = self.calc_elongation(props)
        compactness = self.calc_compactness(texture, props)
        dist_ellipse = DistributionEllipseBuilder().build(props)

        self.pattern_props = pd.DataFrame()
        self.pattern_props['density'] = [density]
        self.pattern_props['elongation'] = [elongation]
        self.pattern_props['orientation'] = [dist_ellipse.orientation]
        self.pattern_props['compactness'] = [compactness]
        self.pattern_props['structural_anisotropy'] = [dist_ellipse.anisotropy]
        self.pattern_props['complexity'] = [np.sum(props['complexity'])]
        return self.pattern_props

    def calc_density(self, tex):
        """
        Calculates the density of objects in the texture.

        Args:
            tex: The texture array.

        Returns:
            float: The density of objects in the texture.
        """
        return np.sum(tex > 0) / tex.size

    def calc_elongation(self, props):
        """
        Calculates the elongation of objects in the texture.

        Args:
            props: The properties of objects in the texture.

        Returns:
            float: The elongation of objects in the texture.
        """
        return np.mean(props['major_axis_length'] / props['minor_axis_length'])

    def calc_compactness(self, tex, props):
        """
        Calculates the compactness of objects in the texture.

        Args:
            tex: The texture array.
            props: The properties of objects in the texture.

        Returns:
            float: The compactness of objects in the texture.
        """
        area_array = props['area']
        quant = np.quantile(area_array, 0.75)
        return 100 * quant / tex.size
