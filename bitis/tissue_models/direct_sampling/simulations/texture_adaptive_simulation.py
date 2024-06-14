from tqdm import tqdm

from bitis.tissue_models.direct_sampling.simulation_path_builder import RandomPathIndexBuilder
from bitis.tissue_models.direct_sampling.training_data_builders import DistanceBuilder
from .texture_adaptive_template_builder import TextureAdaptiveTemplateBuilder


class TextureAdaptiveSimulation:
    """
    Attributes:
        image (numpy.ndarray): The training image.
        out_image (numpy.ndarray): The output image.
        max_size (int): The maximum number exiting neighbours in template.
        max_distance (int): The maximum distance for neighbours search.
        distance_threshold (float): The distance threshold for the distance builder.
        template_sizes (list): The list of template sizes.
    """
    def __init__(self, image, out_image, max_size=20, max_distance=30,
                 min_distance=3, distance_threshold=0.0):
        self.path_builder = RandomPathIndexBuilder(out_image.shape)
        self.template_builder = TextureAdaptiveTemplateBuilder(out_image,
                                                               max_size,
                                                               max_distance,
                                                               min_distance)
        self.distance_builder = DistanceBuilder(image, distance_threshold)
        self.template_sizes = []

    def run(self):
        coords = self.path_builder.build()
        for coord in coords:
            template = self.template_builder.build(*coord)
            closest_pixel = self.distance_builder.build(template)
            self.template_builder.update_image(*coord, closest_pixel)
            self.template_sizes.append(template.shape)
        return self.template_builder.image
