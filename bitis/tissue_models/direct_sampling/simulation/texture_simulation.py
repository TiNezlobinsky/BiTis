import numpy as np

from bitis.tissue_models.direct_sampling.simulation_kernel.sampling_kernel import sampling_kernel
from bitis.tissue_models.direct_sampling.simulation_kernel.labeled_kernel import labeled_kernel


class TextureSimulation:
    def __init__(self):
        self.precondition_matrix = None
        self.labeled_matrix      = None
        self.simulation_path     = None
        self.training_data       = []
        self.trainig_meta        = None
        self.simulation_kernel   = None

        self.scan_faction = 0.8
        self.threshold    = 0.0

    def run(self):
        # raise error if all training_data are not with the same shape
        template_size = self.training_data[0].shape
        pad_i = template_size[0] // 2
        pad_j = template_size[1] // 2

        simulation = np.zeros([self.precondition_matrix.shape[0] + 2*pad_i, 
                               self.precondition_matrix.shape[1] + 2*pad_j])
        simulation[pad_i:-pad_i, pad_j:-pad_j] = self.precondition_matrix

        if self.labeled_matrix:
            labeled_matrix_ = np.zeros([self.precondition_matrix.shape[0] + 2*pad_i, 
                                        self.precondition_matrix.shape[1] + 2*pad_j])
            labeled_matrix_[pad_i:-pad_i, pad_j:-pad_j] = self.labeled_matrix

            return labeled_kernel(simulation, labeled_matrix_, self.simulation_path, self.training_data, 
                                  self.training_meta, pad_i, pad_j, self.scan_faction, self.threshold)
        else:
            return sampling_kernel(simulation, self.simulation_path, self.training_data, pad_i, pad_j, 
                                   self.scan_faction, self.threshold)


