import numpy as np
from scipy import signal
from skimage import transform


class RotatedDistanceBuilder:
    """
    Attributes:
        image (numpy.ndarray): The training image.
        distance_threshold (float): The distance threshold.
    """
    def __init__(self, image, angle_matrix, distance_threshold=0.0):
        self.image              = image
        self.angle_matrix       = angle_matrix
        self.distance_threshold = distance_threshold

        self.rotated_images = {}

        self.calc_rotations(image, angle_matrix)

    def calc_rotations(self, image, angle):
        angles = list(np.unique(self.angle_matrix))
        for angle in angles:
            rotated_image = transform.rotate(image, angle, resize=True, preserve_range=True, order=0)

            non_empty_mask = rotated_image != 0

            non_zero_coords = np.argwhere(non_empty_mask)
            top_left = non_zero_coords.min(axis=0)
            bottom_right = non_zero_coords.max(axis=0) + 1  # Add 1 to include the max row/column

            cropped_image = rotated_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            self.rotated_images[angle] = cropped_image

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

    def calc_min_distance_idx(self, template, image):
        """Calculate the minimum distance index. If minimum distance is less
        than the distance threshold, return a random index.

        Args:
            template (numpy.ndarray): The template.
        """
        if template.sum() == 0:
            x = np.random.randint(image.shape[0])
            y = np.random.randint(image.shape[1])
            return x, y

        distance_map = self.calc_distance_map(image, template)
        distance_threshold = max(distance_map.max(), self.distance_threshold)
        coords = np.argwhere(distance_map >= distance_threshold)

        if len(coords) == 0:
            x = np.random.randint(image.shape[0])
            y = np.random.randint(image.shape[1])
            return x, y

        i = np.random.choice(np.arange(len(coords)))
        x, y = coords[i]
        x += template.shape[0] // 2
        y += template.shape[1] // 2
        return x, y

    def build(self, template, i, j):
        """Calculate the minimum distance index and return the corresponding
        pixel value.

        Args:
            template (numpy.ndarray): The template.

        Returns:
            int: The pixel value.
        """
        angle = self.angle_matrix[i, j]
        image = self.rotated_images[angle]
        x, y = self.calc_min_distance_idx(template, image)
        return image[x, y]
