import numpy as np
from skimage import transform

from bitis.tissue_models.direct_sampling.training_data_builder.training_data_builder import TrainingDataBuilder


class RotatedDataBuilder(TrainingDataBuilder):
    def __init__(self, template_size, training_textures_set, angles_matrix):
        TrainingDataBuilder.__init__(self, template_size, training_textures_set)

        self.angles_matrix = angles_matrix

    def build(self):
        pad_i = self.template[0] // 2
        pad_j = self.template[1] // 2

        angles = list(np.unique(self.angles_matrix))

        meta   = []
        events = []

        start_index = 0
        for angle in angles:
            meta.extend([angle, float(start_index)])

            for texture in self.training_textures_set:
                ni, nj = texture.shape

                rotated = transform.rotate(texture, angle, resize=True, preserve_range=True, order=0)
                events_ = []
                for i in range(ni):
                    for j in range(nj):
                        event = rotated[i:i+2*pad_i, j:j+2*pad_j]
                        if event.shape[0] != 2*pad_i or event.shape[1] != 2*pad_j:
                            continue
                        if 0 not in event:
                            events_.append(event)

                events_ = np.random.permutation(events_)
                events.extend(list(events_))

            meta.append(float(len(events)-1))
            start_index = len(events)

        return np.array(events), meta

