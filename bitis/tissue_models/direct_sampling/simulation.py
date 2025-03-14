import numpy as np
from tqdm import tqdm


class Simulation:
    """
    Attributes:
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
        self.precondition = None
        self.path_builder = None
        self.template_builder = None
        self.template_matching = None

    def run(self, max_iter=None):
        """
        Run simulation.
        """
        if self.precondition is not None:
            self.precondition.apply()

        coords = self.path_builder.build()

        self._index_map = np.zeros_like(self.path_builder.simulation_image)

        if max_iter is not None:
            coords = coords[:max_iter]

        for coord in tqdm(coords):
            template = self.template_builder.build(coord)
            best_match = self.template_matching.run(coord, template)
            self.path_builder.update(coord, best_match)
            self._index_map[*coord] = self.template_matching._best_index

        return self.path_builder.simulation_image
