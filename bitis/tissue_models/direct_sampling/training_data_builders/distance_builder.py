import numpy as np
from scipy import signal

from .adaptive_training_data_builder import AdaptiveTrainingDataBuilder


class DistanceBuilder:
    """
    Attributes:
        image (numpy.ndarray): The training image.
        distance_threshold (float): The distance threshold.
    """
    def __init__(self, image, distance_threshold=0.0):
        self.image = image
        self.distance_threshold = distance_threshold

    def calc_distance_map(self, image, template):
        """Calculate the distance map between the training image and
        the template.

        Args:
            image (numpy.ndarray): The training image.
            template (numpy.ndarray): The template.
        """
        dist_fibr = signal.correlate((image == 2).astype(np.float32),
                                     (template == 2).astype(np.float32),
                                     mode='valid', method='fft')
        dist_myo = signal.correlate((image == 1).astype(np.float32),
                                    (template == 1).astype(np.float32),
                                    mode='valid', method='fft')
        dist = dist_fibr + dist_myo
        return dist / (template > 0).sum()

    def calc_min_distance_idx(self, template, i_shift, j_shift):
        """Calculate the minimum distance index. If minimum distance is less
        than the distance threshold, return a random index.

        Args:
            template (numpy.ndarray): The template.
        """
        if template.sum() == 0:
            x = np.random.randint(self.image.shape[0])
            y = np.random.randint(self.image.shape[1])
            return x, y

        distance_map = self.calc_distance_map(self.image, template)
        distance_threshold = max(distance_map.max(), self.distance_threshold)
        coords = np.argwhere(distance_map >= distance_threshold)

        if len(coords) == 0:
            x = np.random.randint(self.image.shape[0])
            y = np.random.randint(self.image.shape[1])
            return x, y

        i = np.random.choice(np.arange(len(coords)))
        x_, y_ = coords[i]

        x = x_ + i_shift
        y = y_ + j_shift
        return x, y

    def reset_image(self, i, j, image, tr_shape):
        i_min, i_max, j_min, j_max = AdaptiveTrainingDataBuilder.build(
            i, j, image.shape, tr_shape)

        self.image = image[i_min: i_max, j_min: j_max]
        return i_min, j_min

    def build(self, template, i_shift, j_shift):
        """Calculate the minimum distance index and return the corresponding
        pixel value.

        Args:
            template (numpy.ndarray): The template.

        Returns:
            int: The pixel value.
        """
        x, y = self.calc_min_distance_idx(template, i_shift, j_shift)

        if x >= self.image.shape[0] or y >= self.image.shape[1]:
            print('Error: x, y =', x, y)
            print('Error: image shape =', self.image.shape)
            print('Error: template shape =', template.shape)
            print('Error: i_shift, j_shift =', i_shift, j_shift)
        return self.image[x, y]
