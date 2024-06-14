import numpy as np


class TrainingDataBuilder:
    def __init__(self, template_size, training_textures_set):
        """
        Initializes a TrainingDataBuilder object.

        Args:
            template_size (tuple): The size of the template.
            training_textures_set (list): A list of training textures.

        """
        self.template_size = template_size
        self.training_textures_set = training_textures_set

    def build(self):
        """
        Builds the training data.

        Returns:
            numpy.ndarray: The built training data.

        """
        pad_i = self.template_size[0] // 2
        pad_j = self.template_size[1] // 2

        events = []
        for texture in self.training_textures_set:
            ni, nj = texture.shape

            for i in range(ni):
                for j in range(nj):
                    event = texture[i:i+2*pad_i, j:j+2*pad_j]            
                    if event.shape[0] != 2*pad_i or event.shape[1] != 2*pad_j:
                        continue
                    if 0 not in event:
                        events.append(event)

        return np.random.permutation(events)
