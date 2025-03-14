from abc import ABC, abstractmethod


class SimulationPathBuilder(ABC):
    """
    A class that builds simulation paths.
    """

    def __init__(self):
        pass

    @abstractmethod
    def build(self):
        pass

    def update(self, coord, value):
        """
        Update the simulation image with the value at the given coordinate.
        """
        self.simulation_image[*coord] = value
