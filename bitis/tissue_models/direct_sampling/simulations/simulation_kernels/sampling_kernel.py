import numpy as np
from numba import njit


@njit(nogil=True)
def sampling_kernel(simulation, simulation_path, training_data, pad_i, pad_j, progress, scan_fraction=0.5, 
                    threshold=0.0):

    training_index = 0

    sim_path_len = len(simulation_path)
    training_len = len(training_data)

    for simulation_index in range(sim_path_len):
        simulation_node  = simulation_path[simulation_index]
        simulation_event = simulation[simulation_node[0] - pad_i:simulation_node[0] + pad_i,
                                      simulation_node[1] - pad_j:simulation_node[1] + pad_j]

        mindist  = np.inf
        tries    = 0       
        max_scan = sim_path_len*scan_fraction

        while True:
            training_index += 1
            tries += 1

            if training_index > training_len-1:
                training_index = 0

            training_event = training_data[training_index]
            distance = np.mean(simulation_event != training_event)
    
            if distance <= threshold or np.isnan(distance):
                simulation[int(simulation_node[0]), int(simulation_node[1])] = training_event[pad_i, pad_j]
                break
            else:
                if distance < mindist:
                    mindist = distance
                    best_index = training_index
    
                if tries > max_scan:
                    simulation[int(simulation_node[0]), int(simulation_node[1])] = training_data[best_index][pad_i, pad_j]
                    break
        
        progress.update(1)
                        
    simulation = simulation[pad_i:-pad_i, pad_j:-pad_j]

    return simulation

