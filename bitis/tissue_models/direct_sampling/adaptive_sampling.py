from .simulation import Simulation
from .simulation_path import RandomSimulationPathBuilder
from .template_builders import AdaptiveTemplateBuilder
from .template_matching import SingleImageMatching


class AdaptiveSampling(Simulation):
    """
    Direct sampling with adaptive template size. The template size is
    determined by the number of known pixels in the template.

    Attributes:
        simulation_image (numpy.ndarray): The simulation image.
        path_builder (RandomSimulationPathBuilder): The path builder.
        template_builder (AdaptiveTemplateBuilder): The template builder.
        template_matching (SingleImageMatching): The template matching.
    """
    def __init__(self, simulation_image, training_image, max_known_pixels,
                 max_template_size, min_template_size=5, num_of_candidates=1,
                 min_known_pixels=1, tissue_mask=None, use_tf=False):
        """
        Args:
            simulation_image (numpy.ndarray): The simulation image.
            training_image (numpy.ndarray): The training image.
            max_known_pixels (int): The maximum number of known pixels in the
                template.
            max_template_size (int): The maximum size of the template.
            min_template_size (int): The minimum size of the template.
            num_of_candidates (int): The number of candidates to select from.
            min_known_pixels (int): The minimum number of known pixels in the
                template.
            tissue_mask (numpy.ndarray): Boolean mask for tissue pixels.
            use_tf (bool): Whether to use TensorFlow for template matching.
        """
        super().__init__()
        self.path_builder = RandomSimulationPathBuilder(simulation_image,
                                                        tissue_mask)
        self.template_builder = AdaptiveTemplateBuilder(simulation_image,
                                                        max_known_pixels,
                                                        max_template_size,
                                                        min_template_size)
        self.template_matching = SingleImageMatching(training_image,
                                                     num_of_candidates,
                                                     min_known_pixels,
                                                     use_tf=use_tf)
