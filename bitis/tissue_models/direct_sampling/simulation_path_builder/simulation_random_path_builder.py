import numpy as np

from bitis.tissue_models.direct_sampling.simulation_path_builder.simulation_path_builder import SimulationPathBuilder


class SimulationRandomPathBuilder(SimulationPathBuilder):
    def __init__(self):
        SimulationPathBuilder.__init__(self)

    def build(self, template_size, simulation_size):
        pad_i = template_size[0] // 2
        pad_j = template_size[1] // 2
        pad_k = None
        if len(template_size) > 2:
            pad_k = template_size[2] // 2

        ci = np.arange(pad_i, (simulation_size[0] + template_size[0]) - pad_i)
        cj = np.arange(pad_j, (simulation_size[1] + template_size[1]) - pad_j)
        ck = None
        if pad_k:
            ck = np.arange(pad_k, (simulation_size[2] + template_size[2]) - pad_k)
        Ci, Cj = np.meshgrid(ci, cj)
        Ck = None
        if ck:
            Ci, Cj, Ck = np.meshgrid(ci, cj, ck)
        coordinates = np.stack((Ci.ravel(), Cj.ravel()), axis=1)
        if Ck:
            coordinates = np.stack((Ci.ravel(), Cj.ravel(), Ck.ravel()), axis=1)
        return np.random.permutation(coordinates)