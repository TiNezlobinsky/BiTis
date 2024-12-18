import numpy as np
from .simulation_path_builder import SimulationPathBuilder


class RandomSimulationPathBuilder(SimulationPathBuilder):
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
        super().__init__()
        self.simulation_image = simulation_image
        self.tissue_mask = tissue_mask

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
        return coords
