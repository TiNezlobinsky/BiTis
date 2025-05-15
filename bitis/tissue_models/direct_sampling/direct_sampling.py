from .simulation import Simulation
from .simulation_path import PaddedSimulationPathBuilder
from .template_builders import FixedTemplateBuilder
from .template_matching import ImageCollectionMatching


class DirectSampling(Simulation):
    """
    Attributes:
        simulation_image: np.ndarray
            The image that will be created by the simulation
        precondition: Precondition
            The precondition that will be applied to the simulation image
        path_builder: SimulationPathBuilder
            The path builder that will be used to create the path
        template_builder: TemplateBuilder
            The template builder that will be used to create the template
        template_matching: TemplateMatching
            The template matching that will be used to find the closest pixel
    """
    def __init__(self, simulation_image, training_images, template_size=None,
                 template_shape=None, min_distance=.1, max_fraction=0.7):
        super().__init__()
        self.precondition = None
        self.path_builder = PaddedSimulationPathBuilder(simulation_image,
                                                        template_size,
                                                        template_shape)
        self.simulation_image = self.path_builder.simulation_image
        self.template_builder = FixedTemplateBuilder(self.simulation_image,
                                                     template_size,
                                                     template_shape)
        self.template_matching = ImageCollectionMatching(training_images,
                                                         template_size,
                                                         template_shape,
                                                         min_distance,
                                                         max_fraction)

    def run(self):
        """
        Run the direct sampling simulation.
        """
        super().run()
        return self.path_builder.simulated_image
