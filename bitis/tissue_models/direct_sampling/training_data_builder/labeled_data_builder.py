import numpy as np

from bitis.tissue_models.direct_sampling.training_data_builder.training_data_builder import TrainingDataBuilder


class LabeledDataBuilder(TrainingDataBuilder):
    def __init__(self, template_size, training_textures_set, labeled_matrices):
        TrainingDataBuilder.__init__(self, template_size, training_textures_set)

        self.labeled_matrices = labeled_matrices

        # raise error if labeled_matrices length != training_textures_set length

    def build(self):
        pad_i = self.template[0] // 2
        pad_j = self.template[1] // 2
        
        labels = []
        for matrix in self.labeled_matrices:
            labels.extend(list(np.unique(matrix)))

        meta   = []
        events = []

        start_index = 0
        for label in labels:
            meta.extend([label, float(start_index)])

            for idx in len(self.training_textures_set):
                texture = self.training_textures_set[idx]
                matrix  = self.labeled_matrices[idx]
                ni, nj = texture.shape

                events_ = []
                for i in range(ni):
                    for j in range(nj):
                        event   = texture[i:i+2*pad_i, j:j+2*pad_j]
                        labeled = matrix[i:i+2*pad_i, j:j+2*pad_j]
                        if event.shape[0] != 2*pad_i or event.shape[1] != 2*pad_j:
                            continue
                        if 0 not in event:
                            if np.all(labeled == label):
                                events_.append(event)

                events_ = np.random.permutation(events_)
                events.extend(list(events_))

            meta.append(float(len(events)-1))
            start_index = len(events)

        return np.array(events), meta

