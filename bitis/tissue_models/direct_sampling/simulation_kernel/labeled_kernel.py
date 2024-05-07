import numpy as np
from numba import njit


@njit
def labeled_kernel(simulation, labeled_matrix, simulation_path_coordinates, training_data, training_meta,
                    pad_i, pad_j, scan_fraction=0.1, threshold=0.0):
    
    training_indices = np.zeros(np.unique(labeled_matrix).size, dtype=numba.int32)

    for simulation_index in range(len(simulation_path_coordinates)):
        simulation_node  = simulation_path_coordinates[simulation_index]
        simulation_label = labeled_matrix[simulation_node[0], simulation_node[1]]
        simulation_event = simulation[simulation_node[0] - pad_i:simulation_node[0] + pad_i,
                                      simulation_node[1] - pad_j:simulation_node[1] + pad_j]

        labeled_events = None
        curr_i = 0
        for i in range(0, len(training_meta), 3):
            if training_meta[i] == simulation_label:
                labeled_events = training_data[int(training_meta[i+1]):int(training_meta[i+2])]
                break
            curr_i += 1
 
        mindist  = np.inf
        tries    = 0       
        max_scan = len(simulation_path_coordinates)*scan_fraction

        while True:
            training_indices[curr_i] += 1
            tries += 1
            
            if training_indices[curr_i] > len(labeled_events)-1:
                training_indices[curr_i] = 0
            training_event = labeled_events[training_indices[curr_i]]

            distance = np.mean(simulation_event != training_event)
            
            if distance <= threshold or np.isnan(distance):
                simulation[int(simulation_node[0]), int(simulation_node[1])] = training_event[pad_i, pad_j]
                break
            else:
                if distance < mindist:
                    mindist = distance
                    best_index = training_indices[curr_i]
    
                if tries > max_scan:
                    simulation[int(simulation_node[0]), int(simulation_node[1])] =  labeled_events[best_index][pad_i, pad_j]
                    break                    
                        
    simulation = simulation[pad_i:-pad_i, pad_j:-pad_j]

    return simulation

