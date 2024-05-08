import numpy as np


class TextureSimulation:
    def __init__(self):
        self.sim_size = []

        self.precondition_matrix = None
        self.labeled_matrix      = None
        self.simulation_path     = None
        self.training_data       = None
        self.trainig_meta        = None
        self.simulation_kernel   = None

        self.scan_faction = 0.8
        self.threshold    = 0.0

    def run(self):
        # raise error if all training_data are not with the same shape
        template_size = self.training_data.shape
        pad_i = template_size[0] // 2
        pad_j = template_size[1] // 2

        simulation = np.zeros([self.precondition_matrix.shape[0] + 2*pad_i, 
                               self.precondition_matrix.shape[1] + 2*pad_j])
        simulation[pad_i:-pad_i, pad_j:-pad_j] = self.precondition_matrix
        
        ci = np.arange(pad_i, simulation.shape[0] - pad_i)
        cj = np.arange(pad_j, simulation.shape[1] - pad_j)
        Ci, Cj = np.meshgrid(ci, cj)
        coordinates = np.stack((Ci.ravel(), Cj.ravel()), axis=1)
        simulation_path_coordinates = np.random.permutation(coordinates) 

        if angle_matrix is not None:
            transformed_events, transformed_meta = prepare_transformed_dataset(training_image, angle_matrix, template)
            angle_matrix_ = np.zeros([precondition.shape[0] + 2*pad_i, precondition.shape[1] + 2*pad_j])
            angle_matrix_[pad_i:-pad_i, pad_j:-pad_j] = angle_matrix
            angle_matrix = angle_matrix_

            return kernel_transformed(training_image, simulation, angle_matrix, transformed_events, transformed_meta, 
                                    simulation_path_coordinates, pad_i, pad_j, scan_fraction=0.1, threshold=0.0)
        else:
            return kernel_standard(training_image, simulation, simulation_path_coordinates, training_path_coordinates, 
                                    pad_i, pad_j, scan_fraction=0.1, threshold=0.0)      

    

