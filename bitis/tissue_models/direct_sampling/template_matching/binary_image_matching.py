import numpy as np
import tensorflow as tf
from scipy import fft as spfft
from .template_maching import TemplateMatching


class BinaryImageMatching(TemplateMatching):
    """
    Template matching with a single image using the FFT.

    Attributes:
        fft_calc (object): The FFT calculator.
        training_image (numpy.ndarray): The training image.
        n_candidates (int): Number of best matching pixels to consider.
        min_known_pixels (int): Minimum number of known pixels in the template.
    """
    def __init__(self, training_image, n_candidates=1, min_known_pixels=1):
        """
        Args:
            training_image (numpy.ndarray): The training image.
            n_candidates (int): Number of best matching pixels to consider.
            min_known_pixels (int): Minimum number of known pixels in the
                template.
        """
        super().__init__()
        self.fft_calc = SPFFT()
        self.training_image = training_image
        self.n_candidates = n_candidates
        self.min_known_pixels = min_known_pixels
        self._best_index = -1
        self._template_size = []

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
        self.fft_shape = [spfft.next_fast_len(s, True) for s in image.shape]
        self.fft_image = self.fft_calc.rfftnd(image, self.fft_shape)

    def run(self, coord, template_args):
        """Search for the best matching pixel in the training image.

        Args:
            coord (tuple): The coordinates of the target pixel.
            template_args (tuple): The output of the template builder. The
                first element is the template and the second element is the
                coordinates of the target pixel on the template.
        """
        template, coord_on_template = template_args
        if self.n_candidates < 1:
            raise ValueError("Number of candidates must be greater than 0.")

        if np.count_nonzero(template) < self.min_known_pixels:
            return self.random_pixel()

        distance_map = self.compute_distance_map(template)
        best_coord = self.find_best_match(distance_map, coord_on_template)
        return self.training_image[*best_coord]

    def find_best_match(self, distance_map, coord_on_template):
        """Match the template to the training image.

        Args:
            distance_map (numpy.ndarray): The distance map.
            coord_on_template (tuple): The coordinates of the target pixel on
                the template.

        Returns:
            int: The pixel value.
        """

        if self.n_candidates == 1:
            best_coord = self.best_match_coord(distance_map)

        if self.n_candidates > 1:
            best_coord = self.best_candidate_coord(distance_map,
                                                   self.n_candidates)

        best_coord = [c + t for c, t in zip(best_coord, coord_on_template)]
        self._best_index = np.ravel_multi_index(best_coord,
                                                self.training_image.shape)
        return best_coord

    def random_pixel(self):
        """Return a random pixel from the training image."""
        coord = [np.random.randint(i) for i in self.training_image.shape]

        self._best_index = np.ravel_multi_index(coord,
                                                self.training_image.shape)

        return self.training_image[*coord]

    def best_match_coord(self, distance_map):
        """Find the best matching pixel.

        Args:
            distance_map (numpy.ndarray): The distance map.

        Returns:
            tuple: The coordinates of the best matching pixel.
        """
        best_inds = distance_map.argmin()
        coord = np.unravel_index(best_inds, distance_map.shape)
        return coord

    def best_candidate_coord(self, distance_map, n_candidates):
        """Select pixel among the best candidates.

        Args:
            distance_map (numpy.ndarray): The distance map.
            n_candidates (int): Number of best matching pixels to
                consider.

        Returns:
            tuple: The coordinates of the selected pixel.
        """
        coords = self.select_candidates(distance_map, n_candidates)
        random_ind = np.random.randint(self.n_candidates)
        coord = [c[random_ind] for c in coords]
        return coord

    def select_candidates(self, distance_map, n_candidates):
        """Select the best candidates. The method uses a binary search to speed
        up the process. The candidate search starts in range of the minimum and
        minimum + (maximum - minimum) / 256. If the number of candidates is not
        reached, the range is increased by a factor of 2 and the search is
        repeated.

        Args:
            distance_map (numpy.ndarray): The distance map.
            n_candidates (int): Number of best matching pixels to
                consider.

        Returns:
            tuple: The coordinates of the best candidates.
        """
        min_value = distance_map.min()
        max_value = distance_map.max()
        n = 256

        threshold = min_value + (max_value - min_value) / n
        distances = distance_map.ravel()

        mask = distances < threshold
        while distances[mask].size < n_candidates:
            n //= 2
            threshold = min_value + (max_value - min_value) / n
            mask = distances < threshold

        inds = np.where(mask)[0]
        candidates = distances[mask]

        if len(candidates) == n_candidates:
            return np.unravel_index(inds, distance_map.shape)

        inds_ = np.argpartition(candidates, n_candidates)[:n_candidates]
        coords = np.unravel_index(inds[inds_], distance_map.shape)
        return coords

    def compute_distance_map(self, template):
        """Calculate the distance map between the training image and
        the template as the fraction of matching pixels.

        Args:
            template (numpy.ndarray): The template.
        """
        template = template.copy()
        template[template == 2] = -1

        fft_template = self.fft_calc.rfftnd(template, self.fft_shape)
        matching_pixels = self.fft_calc.irfftnd(
            self.fft_calc.multiply(self.fft_image, fft_template),
            self.fft_shape
        )

        slices = [slice(0, s_tr - s_te + 1)
                  for s_tr, s_te in zip(self.training_image.shape,
                                        template.shape)]
        matching_pixels = matching_pixels[tuple(slices)]
        known_pixels = np.count_nonzero(template != 0)
        matching_pixels = 0.5 * (matching_pixels + known_pixels)
        return 1 - matching_pixels / known_pixels


class TFFFT:
    def __init__(self):
        pass

    def rfftnd(self, image, shape, workers=None):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        if len(shape) == 2:
            return tf.signal.rfft2d(image, shape)
        if len(shape) == 3:
            return tf.signal.rfft3d(image, shape)
        raise ValueError("Only 2D and 3D images are supported.")

    def irfftnd(self, image, shape, workers=None):
        if len(shape) == 2:
            return tf.math.real(tf.signal.irfft2d(image, shape)).numpy()
        if len(shape) == 3:
            return tf.math.real(tf.signal.irfft3d(image, shape)).numpy()
        raise ValueError("Only 2D and 3D images are supported.")

    def multiply(self, image1, image2):
        return image1 * tf.math.conj(image2)


class SPFFT:
    def __init__(self):
        pass

    def rfftnd(self, image, shape, workers=None):
        return spfft.rfftn(image, s=shape, workers=workers)

    def irfftnd(self, image, shape, workers=None):
        return spfft.irfftn(image, shape, workers=workers).real

    def multiply(self, image1, image2):
        return image1 * image2.conj()
