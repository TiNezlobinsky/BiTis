from tqdm import tqdm


class Simulation:
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
    def __init__(self):
        self.simulation_image = None
        self.precondition = None
        self.path_builder = None
        self.template_builder = None
        self.template_matching = None
        # self.template_shapes = []

    def run(self):
        """
        Run simulation.
        """
        if self.precondition is not None:
            self.precondition.apply()

        coords = self.path_builder.build()
        self.simulation_image = self.path_builder.simulation_image

        for coord in tqdm(coords):
            template, coord_on_template = self.template_builder.build(coord)
            closest_pixel = self.template_matching.run(template,
                                                       coord_on_template)
            self.simulation_image[*coord] = closest_pixel
            # self.template_shapes.append(template.shape)

        return self.simulation_image
