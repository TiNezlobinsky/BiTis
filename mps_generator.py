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
        meta.extend([start_index, angle])
        tr = transform.rotate(training_image, angle, resize=True, preserve_range=True, order=0)
        ni, nj = tr.shape
        for i in range(ni):
            for j in range(nj):
                event = tr[i:i+2*pad_i, j:j+2*pad_j]
                if event.shape[0] != 2*pad_i or event.shape[1] != 2*pad_j:
                    continue
                if 0 not in event:
                    events.append(event)
        meta.append(len(events)-1)
        start_index = len(events)

    return np.array(events), meta
        

def simulation(training_image, precondition, template, angle_matrix, scan_fraction=0.1, threshold=0.0):
    pad_i = template[0] // 2
    pad_j = template[1] // 2

    transformed_events, transformed_meta = prepare_transformed_dataset(training_image, angle_matrix, template)

    simulation = np.zeros([precondition.shape[0] + 2*pad_i, precondition.shape[1] + 2*pad_j])
    simulation[pad_i:-pad_i, pad_j:-pad_j] = precondition

    angle_matrix_ = np.zeros([precondition.shape[0] + 2*pad_i, precondition.shape[1] + 2*pad_j])
    angle_matrix_[pad_i:-pad_i, pad_j:-pad_j] = angle_matrix
    angle_matrix = angle_matrix_
    
    ci = np.arange(pad_i, simulation.shape[0]+1 - pad_i)
    cj = np.arange(pad_j, simulation.shape[1]+1 - pad_j)
    Ci, Cj = np.meshgrid(ci, cj)
    coordinates = np.stack((Ci.ravel(), Cj.ravel()), axis=1)
    simulation_path_coordinates = np.random.permutation(coordinates) 
    
    ci = np.arange(pad_i, training_image.shape[0]+1 - pad_i)
    cj = np.arange(pad_j, training_image.shape[1]+1 - pad_j)
    Ci, Cj = np.meshgrid(ci, cj)
    coordinates = np.stack((Ci.ravel(), Cj.ravel()), axis=1)
    training_path_coordinates = np.random.permutation(coordinates) 

    transformed_events = np.random.permutation(transformed_events) 

    return simulation_kernel(training_image, simulation, transformed_events, simulation_path_coordinates, training_path_coordinates, 
                             pad_i, pad_j, scan_fraction=0.1, threshold=0.0)
 

@njit
def simulation_kernel(training_image, simulation, transformed_events, simulation_path_coordinates, training_path_coordinates, 
                      pad_i, pad_j, scan_fraction=0.1, threshold=0.0):

    training_index = 0

    for simulation_index in range(len(simulation_path_coordinates)):
        # print (simulation_index)
        simulation_node = simulation_path_coordinates[simulation_index]
        simulation_event = simulation[simulation_node[0] - pad_i:simulation_node[0] + pad_i,
                                      simulation_node[1] - pad_j:simulation_node[1] + pad_j]

        mindist  = np.inf
        tries    = 0       
        max_scan = len(simulation_path_coordinates)*scan_fraction # max number of scans

        while True:
            training_index += 1
            tries += 1
            
            if transformed_events is not None:
                if training_index > len(transformed_events)-1:
                    training_index = 0
                training_event = transformed_events[training_index]

                distance = np.mean(simulation_event != training_event)
                
                if distance <= threshold or np.isnan(distance):
                    simulation[int(simulation_node[0]), int(simulation_node[1])] = training_event[pad_i, pad_j]
                    break
                else:
                    if distance < mindist:
                        mindist = distance
                        best_index = training_index
        
                    if tries > max_scan:
                        simulation[int(simulation_node[0]), int(simulation_node[1])] =  transformed_events[best_index][pad_i, pad_j]
                        break                    
            else:
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