import numpy as np
import matplotlib.pyplot as plt

from bitis.texture.texture import Texture
from bitis.texture.properties_builders.pattern_properties_builder import PatternPropertiesBuilder


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


original_tex = np.load("../data/example_texture.npy")

pattern_builder = PatternPropertiesBuilder()
pattern_properties = pattern_builder.build(original_tex)

texture = Texture()
texture.matrix = original_tex
texture.properties["pattern"] = pattern_properties

print (texture.properties["pattern"])