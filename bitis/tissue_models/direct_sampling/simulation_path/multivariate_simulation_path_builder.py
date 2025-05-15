import numpy as np
from .random_simulation_path_builder import RandomSimulationPathBuilder


class MultivariateSimulationPathBuilder(RandomSimulationPathBuilder):
    """
    Attributes:
        simulation_image (numpy.ndarray): The simulation image.
    """
    def __init__(self, simulation_image, tissue_mask=None):
        """
        Args:
            simulation_image (numpy.ndarray): The simulation image.
            tissue_mask (numpy.ndarray): Boolean mask for tissue pixels.
        """
        super().__init__(simulation_image, tissue_mask)
        self.joint_image = np.zeros_like(simulation_image)

    def build(self):
        """
        Builds a random path for simulation where the value are 0.

        Returns:
            numpy.ndarray: A random permutation of coordinates for
                the simulation path.
        """
        unknown_pixels_mask = self.simulation_image == 0

        if self.tissue_mask is not None:
            unknown_pixels_mask = unknown_pixels_mask & self.tissue_mask

        coords = np.argwhere(unknown_pixels_mask)
        np.random.shuffle(coords)
        self.coords = coords
        return self.coords

    def update(self, coord, values):
        """
        Update the simulation image with the value at the given coordinate.
        """
        self.simulation_image[*coord] = values[0]
        self.joint_image[*coord] = values[1]
