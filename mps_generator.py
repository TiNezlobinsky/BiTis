import numpy as np
from numba import njit
from skimage import transform


def prepare_transformed_dataset(training_image, angle_matrix, template):
    pad_i = template[0] // 2
    pad_j = template[1] // 2
    
    angles = np.unique(angle_matrix)

    meta   = []
    events = []

    start_index = 0
    for angle in angles:
        meta.extend([angle, float(start_index)])
        
        tr = transform.rotate(training_image, angle, resize=True, preserve_range=True, order=0)
        ni, nj = tr.shape
        events_ = []
        for i in range(ni):
            for j in range(nj):
                event = tr[i:i+2*pad_i, j:j+2*pad_j]
                if event.shape[0] != 2*pad_i or event.shape[1] != 2*pad_j:
                    continue
                if 0 not in event:
                    events_.append(event)
        events_ = np.random.permutation(events_)
        events.extend(list(events_))
        
        meta.append(float(len(events)-1))
        start_index = len(events)

    return np.array(events), meta
        

def simulation(training_image, precondition, template, angle_matrix=None, scan_fraction=0.1, threshold=0.0):
    pad_i = template[0] // 2
    pad_j = template[1] // 2

    simulation = np.zeros([precondition.shape[0] + 2*pad_i, precondition.shape[1] + 2*pad_j])
    simulation[pad_i:-pad_i, pad_j:-pad_j] = precondition
    
    ci = np.arange(pad_i, simulation.shape[0] - pad_i)
    cj = np.arange(pad_j, simulation.shape[1] - pad_j)
    Ci, Cj = np.meshgrid(ci, cj)
    coordinates = np.stack((Ci.ravel(), Cj.ravel()), axis=1)
    simulation_path_coordinates = np.random.permutation(coordinates) 
    
    ci = np.arange(pad_i, training_image.shape[0] - pad_i)
    cj = np.arange(pad_j, training_image.shape[1] - pad_j)
    Ci, Cj = np.meshgrid(ci, cj)
    coordinates = np.stack((Ci.ravel(), Cj.ravel()), axis=1)
    training_path_coordinates = np.random.permutation(coordinates) 

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
 

@njit
def kernel_standard(training_image, simulation, simulation_path_coordinates, training_path_coordinates, 
                    pad_i, pad_j, scan_fraction=0.1, threshold=0.0):

    training_index = 0

    for simulation_index in range(len(simulation_path_coordinates)):
        simulation_node  = simulation_path_coordinates[simulation_index]
        simulation_event = simulation[simulation_node[0] - pad_i:simulation_node[0] + pad_i,
                                      simulation_node[1] - pad_j:simulation_node[1] + pad_j]

        mindist  = np.inf
        tries    = 0       
        max_scan = len(simulation_path_coordinates)*scan_fraction

        while True:
            training_index += 1
            tries += 1

            if training_index > len(training_path_coordinates)-1:
                training_index = 0

            training_node = training_path_coordinates[training_index]
            training_event = training_image[training_node[0] - pad_i:training_node[0] + pad_i, 
                                            training_node[1] - pad_j:training_node[1] + pad_j]

            distance = np.mean(simulation_event != training_event)
    
            if distance <= threshold or np.isnan(distance):
                simulation[int(simulation_node[0]), int(simulation_node[1])] = training_image[int(training_node[0]), int(training_node[1])]
                break
            else:
                if distance < mindist:
                    mindist = distance
                    bestpoint = training_node
    
                if tries > max_scan:
                    simulation[int(simulation_node[0]), int(simulation_node[1])] = training_image[int(bestpoint[0]), int(bestpoint[1])]
                    break
                        
    simulation = simulation[pad_i:-pad_i, pad_j:-pad_j]

    return simulation


@njit
def kernel_transformed(training_image, simulation, angle_matrix, transformed_events, transformed_meta, simulation_path_coordinates, 
                      pad_i, pad_j, scan_fraction=0.1, threshold=0.0):

    training_indices = np.zeros(np.unique(angle_matrix).size, dtype=numba.int32)

    for simulation_index in range(len(simulation_path_coordinates)):
        simulation_node  = simulation_path_coordinates[simulation_index]
        simulation_angle = angle_matrix[simulation_node[0], simulation_node[1]]
        simulation_event = simulation[simulation_node[0] - pad_i:simulation_node[0] + pad_i,
                                      simulation_node[1] - pad_j:simulation_node[1] + pad_j]

        angle_events = None
        curr_i = 0
        for i in range(0, len(transformed_meta), 3):
            if transformed_meta[i] == simulation_angle:
                angle_events = transformed_events[int(transformed_meta[i+1]):int(transformed_meta[i+2])]
                break
            curr_i += 1
 
        mindist  = np.inf
        tries    = 0       
        max_scan = len(simulation_path_coordinates)*scan_fraction # max number of scans

        while True:
            training_indices[curr_i] += 1
            tries += 1
            
            if training_indices[curr_i] > len(angle_events)-1:
                training_indices[curr_i] = 0
            training_event = angle_events[training_indices[curr_i]]

            distance = np.mean(simulation_event != training_event)
            
            if distance <= threshold or np.isnan(distance):
                simulation[int(simulation_node[0]), int(simulation_node[1])] = training_event[pad_i, pad_j]
                break
            else:
                if distance < mindist:
                    mindist = distance
                    best_index = training_indices[curr_i]
    
                if tries > max_scan:
                    simulation[int(simulation_node[0]), int(simulation_node[1])] =  angle_events[best_index][pad_i, pad_j]
                    break                    
                        
    simulation = simulation[pad_i:-pad_i, pad_j:-pad_j]

    return simulation
    