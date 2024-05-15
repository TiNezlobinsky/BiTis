import numpy as np
import matplotlib.pyplot as plt

from bitis.texture.texture import Texture
from bitis.texture.properties_builders.pattern_properties_builder import PatternPropertiesBuilder


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


im = plt.imread("../../../MPS_generator/original_texs/or_tex_34.png")
gim = rgb2gray(im)
nim = np.where(gim > 0.5, 1, 2)

pattern_builder = PatternPropertiesBuilder()
pattern_properties = pattern_builder.build(nim)

texture = Texture()
texture.matrix = nim
texture.properties["pattern"] = pattern_properties

print (texture.properties["pattern"])