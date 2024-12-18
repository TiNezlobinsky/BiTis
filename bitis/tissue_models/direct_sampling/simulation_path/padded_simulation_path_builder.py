import numpy as np
from .simulation_path_builder import SimulationPathBuilder


class PaddedSimulationPathBuilder(SimulationPathBuilder):
    """
    Attributes:
        simulation_image (numpy.ndarray): The simulation image.
        template_shape (tuple): The shape of the template.
    """

    def __init__(self, simulation_image, template_size=None,
                 template_shape=None):
        """
        Args:
            simulation_image (numpy.ndarray): The simulation image.
            template_size (int): The size of the template.
            template_shape (tuple): The shape of the template.
        """
        super().__init__()
        self.template_shape = None
        self._simulation_image = simulation_image

        if template_size is not None:
            self.template_size = template_size

        if template_shape is not None:
            self.template_shape = template_shape

        self.simulation_image = simulation_image

    @property
    def simulation_image(self):
        return self._simulation_image

    @simulation_image.setter
    def simulation_image(self, value):
        pad_width = []
        for i in range(len(value.shape)):
            pad_width.append((self.template_shape[i] // 2,
                              self.template_shape[i] -
                              self.template_shape[i] // 2))

        self._simulation_image = np.pad(value, pad_width, mode='constant')

    @property
    def simulated_image(self):
        slices = []
        for i in range(len(self.simulation_image.shape)):
            slices.append(slice(self.template_shape[i] // 2,
                                self.simulation_image.shape[i] -
                                self.template_shape[i] +
                                self.template_shape[i] // 2))
        return self.simulation_image[tuple(slices)]

    @property
    def template_size(self):
        if np.all(self.template_shape != self.template_shape[0]):
            raise ValueError("Template shape is not uniform.")

        return self.template_shape[0]

    @template_size.setter
    def template_size(self, value):
        self.template_shape = (value, ) * len(self.simulation_image.shape)

    def build(self):
        """
        Builds a random path for simulation where the value are 0. If the
        ``template_shape`` is provided, the path will be with padding.

        Returns:
            numpy.ndarray: A random permutation of coordinates for
                the simulation path.
        """

        coords = np.argwhere(self.simulation_image == 0)

        mask = (coords >= np.array(self.template_shape) // 2).all(axis=1)
        mask &= (coords < (np.array(self.simulation_image.shape) -
                           np.array(self.template_shape) +
                           np.array(self.template_shape) // 2)).all(axis=1)
        coords = coords[mask]
        np.random.shuffle(coords)
        return coords
