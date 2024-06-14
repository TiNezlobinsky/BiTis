import numpy as np
from scipy import signal


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
        dist_fibr = signal.correlate((image == 2).astype(float),
                                     (template == 2).astype(float),
                                     mode='valid', method='fft')
        dist_myo = signal.correlate((image == 1).astype(float),
                                    (template == 1).astype(float),
                                    mode='valid', method='fft')
        dist = dist_fibr + dist_myo
        return dist / (template > 0).sum()

    def calc_min_distance_idx(self, template):
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
        x, y = coords[i]
        x += template.shape[0] // 2
        y += template.shape[1] // 2
        return x, y

    def build(self, template):
        """Calculate the minimum distance index and return the corresponding
        pixel value.

        Args:
            template (numpy.ndarray): The template.

        Returns:
            int: The pixel value.
        """
        x, y = self.calc_min_distance_idx(template)
        return self.image[x, y]
