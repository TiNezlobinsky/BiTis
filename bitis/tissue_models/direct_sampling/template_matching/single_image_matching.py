import numpy as np
from scipy import fft
from .template_maching import TemplateMatching


class SingleImageMatching(TemplateMatching):
    """
    Template matching with a single image using the FFT.

    Attributes:
        training_image (numpy.ndarray): The training image.
        min_distance (float): The minimum distance threshold. If the minimum
            distance is more than this value, a random pixel is returned.
    """
    def __init__(self, training_image, min_distance=1.0):
        """
        Args:
            training_image (numpy.ndarray): The training image.
            min_distance (float): The minimum distance threshold. If the
                minimum distance is more than this value, a random pixel is
                returned.
        """
        super().__init__()
        self.training_image = training_image
        self.min_distance = min_distance

    @property
    def training_image(self):
        return self._original_image

    @training_image.setter
    def training_image(self, image):
        self._original_image = image
        # Normalize the image
        image = image.copy().astype(np.float32)
        image[image == 2] = -1
        # Precompute the FFT of the image
        self.fft_shape = [fft.next_fast_len(s, True) for s in image.shape]
        self.fft_image = fft.rfftn(image, s=self.fft_shape)

    def run(self, template, coord_on_template):
        """Calculate the minimum distance index and return the corresponding
        pixel value.

        Args:
            template (numpy.ndarray): The template.

        Returns:
            int: The pixel value.
        """
        if np.count_nonzero(template) == 0:
            return self.random_pixel()

        return self.best_matching_pixel(template, coord_on_template)

    def best_matching_pixel(self, template, coord_on_template):
        """Calculate the minimum distance index. If minimum distance is less
        than the distance threshold, return a random index. If multiple indices
        have the same minimum distance, choose one randomly.

        Args:
            template (numpy.ndarray): The template.
            coord_on_template (tuple): The coordinates of the pixel on the
                template.
        """
        distance_map = self.calc_distance_map(template)

        threshold = min(distance_map.min(), self.min_distance)
        inds = np.flatnonzero(distance_map <= threshold)

        if len(inds) == 0:
            return self.random_pixel()

        random_ind = np.random.choice(inds)
        coord = np.unravel_index(random_ind, distance_map.shape)

        coord = [c + t for c, t in zip(coord, coord_on_template)]
        return self.training_image[*coord]

    def random_pixel(self):
        """Return a random pixel from the training image."""
        coord = [np.random.randint(i) for i in self.training_image.shape]
        return self.training_image[*coord]

    def calc_distance_map(self, template):
        """Calculate the distance map between the training image and
        the template.

        Args:
            template (numpy.ndarray): The template.
        """
        template = template.copy()
        template[template == 2] = -1
        fft_template = fft.rfftn(template, s=self.fft_shape).conj()
        matching_pixels = fft.irfftn(self.fft_image * fft_template).real

        i_max = self.training_image.shape[0] - template.shape[0] + 1
        j_max = self.training_image.shape[1] - template.shape[1] + 1
        matching_pixels = matching_pixels[:i_max, :j_max]

        known_pixels = np.count_nonzero(template != 0)
        matching_pixels = 0.5 * (matching_pixels + known_pixels)
        return 1 - matching_pixels / known_pixels
