import numpy as np
import tensorflow as tf
from scipy import fft as spfft
from .template_maching import TemplateMatching


class SingleImageMatching(TemplateMatching):
    """
    Template matching with a single image using the FFT.

    Attributes:
        fft_calc (object): The FFT calculator.
        training_image (numpy.ndarray): The training image.
        num_of_candidates (int): Number of best matching pixels to consider.
        min_known_pixels (int): Minimum number of known pixels in the template.
    """
    def __init__(self, training_image, num_of_candidates=1, min_known_pixels=1,
                 use_tf=False):
        """
        Args:
            training_image (numpy.ndarray): The training image.
            num_of_candidates (int): Number of best matching pixels to
                consider.
            min_known_pixels (int): Minimum number of known pixels in the
                template.
            use_tf (bool): Use TensorFlow for the FFT calculation.
        """
        super().__init__()
        self.fft_calc = SPFFT()
        if use_tf:
            self.fft_calc = TFFFT()
        self.training_image = training_image
        self.num_of_candidates = num_of_candidates
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

    def run(self, template, coord_on_template):
        """Calculate the minimum distance index and return the corresponding
        pixel value.

        Args:
            template (numpy.ndarray): The template.
            coord_on_template (tuple): The coordinates of the target pixel on
                the template.

        Returns:
            int: The pixel value.
        """
        if np.count_nonzero(template) < self.min_known_pixels:
            return self.random_pixel()

        return self.best_matching_pixel(template, coord_on_template)

    def best_matching_pixel(self, template, coord_on_template):
        """Calculate the minimum distance index. If multiple indices
        have the same minimum distance, choose one randomly.

        Args:
            template (numpy.ndarray): The template.
            coord_on_template (tuple): The coordinates of the pixel on the
                template.
        """
        distance_map = self.compute_distance_map(template, coord_on_template)

        if self.num_of_candidates < 1:
            raise ValueError("The number of candidates must be greater than 0")

        if self.num_of_candidates == 1:
            best_inds = distance_map.argmin()
            coord = np.unravel_index(best_inds, distance_map.shape)
            coord = [c + t for c, t in zip(coord, coord_on_template)]
            self._best_index = np.ravel_multi_index(coord, self.training_image.shape)
            return self.training_image[*coord]

        # inds = np.argpartition(distance_map.ravel(), self.num_of_candidates)
        # inds = np.unravel_index(inds[:self.num_of_candidates],
        #                         distance_map.shape)
        coords = self.closest_pixels(distance_map, self.num_of_candidates)
        random_ind = np.random.randint(self.num_of_candidates)

        self._best_index = distance_map[coords[0][random_ind],
                                        coords[1][random_ind]]

        coord = [c[random_ind] + t for c, t in zip(coords, coord_on_template)]
        # self._template_size.append(template.shape)
        # self._best_index = np.ravel_multi_index(coord, self.training_image.shape)
        return self.training_image[*coord]

    def closest_pixels(self, distance_map, num_of_candidates):
        """Calculate the minimum distance index. If multiple indices
        have the same minimum distance, choose one randomly.

        Args:
            distance_map (numpy.ndarray): The distance map.
            num_of_candidates (int): Number of best matching pixels to
                consider.
        """
        min_value = distance_map.min()
        max_value = distance_map.max()
        n = 256

        median = min_value + (max_value - min_value) / n
        distances = distance_map.ravel()

        mask = distances < median
        while distances[mask].size < num_of_candidates:
            n //= 2
            median = min_value + (max_value - min_value) / n
            mask = distances < median

        inds = np.where(mask)[0]
        candidates = distances[mask]
        if len(candidates) == num_of_candidates:
            return np.unravel_index(inds, distance_map.shape)

        inds_ = np.argpartition(candidates,
                                num_of_candidates)[:num_of_candidates]
        return np.unravel_index(inds[inds_], distance_map.shape)

    def random_pixel(self):
        """Return a random pixel from the training image."""
        coord = [np.random.randint(i) for i in self.training_image.shape]

        # self._best_index = np.ravel_multi_index(coord, self.training_image.shape)

        return self.training_image[*coord]

    def compute_distance_map(self, template, coord_on_template):
        """Calculate the distance map between the training image and
        the template.

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
