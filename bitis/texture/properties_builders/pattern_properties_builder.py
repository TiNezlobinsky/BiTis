
from bitis.texture.properties.pattern_properties import PatternProperties
from bitis.texture.analysis.objects_analysis import ObjectAnalysis
from bitis.texture.properties_builders.properties_builder import PropertiesBuilder


class PatternPropertiesBuilder(PropertiesBuilder):
    """
    A class that builds pattern properties for a given texture.

    Attributes:
        area_threshold (int): The threshold value for the area of objects in the texture.
    """

    def __init__(self):
        PropertiesBuilder.__init__(self)
        self.area_threshold = 5

    def build(self, texture):
        """
        Builds pattern properties for the given texture.

        Args:
            texture: The texture to analyze.

        Returns:
            PatternProperties: The pattern properties of the texture.
        """
        pattern_properties = PatternProperties()

        density, elongation, orientation, compactness = ObjectAnalysis().get_pattern(texture)

        pattern_properties.density = density
        pattern_properties.elongation = elongation
        pattern_properties.orientation = orientation
        pattern_properties.compactness = compactness

        return pattern_properties
    


    