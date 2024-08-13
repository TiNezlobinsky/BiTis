import numpy as np

from bitis.tissue_models.direct_sampling.simulation_path_builder.simulation_path_builder import SimulationPathBuilder


class SimulationRandomPathBuilder(SimulationPathBuilder):
    def __init__(self, template_size=None, simulation_size=None):
        SimulationPathBuilder.__init__(self)
        self.template_size = template_size
        self.simulation_size = simulation_size

    def build(self):
        """
        Builds a random path for simulation based on the given template size and simulation size.

        Args:
            template_size (tuple): The size of the template.
            simulation_size (tuple): The size of the simulation.

        Returns:
            numpy.ndarray: A random permutation of coordinates for the simulation path.
        """
        ci = np.arange(self.simulation_size[0])
        cj = np.arange(self.simulation_size[1])

        ck = None
        if len(self.template_size) > 2:
            ck = np.arange(self.simulation_size[2])
        Ci, Cj = np.meshgrid(ci, cj)
        
        Ck = None
        if ck:
            Ci, Cj, Ck = np.meshgrid(ci, cj, ck)
        coordinates = np.stack((Ci.ravel(), Cj.ravel()), axis=1)
        if Ck:
            coordinates = np.stack((Ci.ravel(), Cj.ravel(), Ck.ravel()), axis=1)

        return np.random.permutation(coordinates)