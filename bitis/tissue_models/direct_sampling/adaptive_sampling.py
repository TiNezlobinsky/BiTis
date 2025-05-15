from .simulation import Simulation
from .simulation_path import RandomSimulationPathBuilder
from .template_builders import AdaptiveTemplateBuilder
from .template_matching import BinaryImageMatching


class AdaptiveSampling(Simulation):
    """
    Direct sampling with adaptive template size. The template size is
    determined by the number of known pixels in the template.

    Attributes:
        simulation_image (numpy.ndarray): The simulation image.
        path_builder (RandomSimulationPathBuilder): The path builder.
        template_builder (AdaptiveTemplateBuilder): The template builder.
        template_matching (BinaryImageMatching): The template matching.
    """
    def __init__(self, simulation_image, training_image, max_known_pixels,
                 max_template_size, min_template_size=5, n_candidates=1,
                 min_known_pixels=1, tissue_mask=None):
        """
        Args:
            simulation_image (numpy.ndarray): The simulation image.
            training_image (numpy.ndarray): The training image.
            max_known_pixels (int): The maximum number of known pixels in the
                template.
            max_template_size (int): The maximum size of the template.
            min_template_size (int): The minimum size of the template.
            n_candidates (int): The number of candidates to select from.
            min_known_pixels (int): The minimum number of known pixels in the
                template.
            tissue_mask (numpy.ndarray): Boolean mask for tissue pixels.
        """
        super().__init__()
        self.path_builder = RandomSimulationPathBuilder(simulation_image,
                                                        tissue_mask)
        self.template_builder = AdaptiveTemplateBuilder(simulation_image,
                                                        max_known_pixels,
                                                        max_template_size,
                                                        min_template_size)
        self.template_matching = BinaryImageMatching(training_image,
                                                     n_candidates,
                                                     min_known_pixels)
