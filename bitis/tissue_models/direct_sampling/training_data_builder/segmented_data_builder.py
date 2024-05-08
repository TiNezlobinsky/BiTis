import numpy as np

from bitis.tissue_models.direct_sampling.training_data_builder.training_data_builder import TrainingDataBuilder


class SegmentedDataBuilder(TrainingDataBuilder):
    def __init__(self, template_size, training_textures_set, segmented_matrices):
        TrainingDataBuilder.__init__(self, template_size, training_textures_set)

        self.segmented_matrices = segmented_matrices

        # raise error if labeled_matrices length != training_textures_set length

    def build(self):
        pad_i = self.template[0] // 2
        pad_j = self.template[1] // 2
        
        segments = []
        for matrix in self.segmented_matrices:
            segments.extend(list(np.unique(matrix)))

        meta   = []
        events = []

        start_index = 0
        for segment in segments:
            meta.extend([segment, float(start_index)])

            for idx in len(self.training_textures_set):
                texture = self.training_textures_set[idx]
                matrix  = self.segmented_matrices[idx]
                ni, nj = texture.shape

                events_ = []
                for i in range(ni):
                    for j in range(nj):
                        event   = texture[i:i+2*pad_i, j:j+2*pad_j]
                        labeled = matrix[i:i+2*pad_i, j:j+2*pad_j]
                        if event.shape[0] != 2*pad_i or event.shape[1] != 2*pad_j:
                            continue
                        if 0 not in event:
                            if np.all(labeled == segment):
                                events_.append(event)

                events_ = np.random.permutation(events_)
                events.extend(list(events_))

            meta.append(float(len(events)-1))
            start_index = len(events)

        return np.array(events), meta

