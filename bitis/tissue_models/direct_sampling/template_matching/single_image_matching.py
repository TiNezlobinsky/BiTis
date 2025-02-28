import numpy as np
import tensorflow as tf
from scipy import fft as spfft
from .template_maching import TemplateMatching


class SingleImageMatching(TemplateMatching):
    """
    Template matching with a single image using the FFT.

    Attributes:
        training_image (numpy.ndarray): The training image.
        min_distance (float): The minimum distance threshold. By default,
            it is set to 0, i.e., the closest pixel is always chosen.
    """
    def __init__(self, training_image, min_distance=.0):
        """
        Args:
            training_image (numpy.ndarray): The training image.
            min_distance (float): The minimum distance threshold. Values should
                be in the range [0, 1]. Defaults to 0., i.e., the closest pixel
                is always chosen.
        """
        super().__init__()
        self.fft_calc = SPFFT()
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
        self.fft_shape = [spfft.next_fast_len(s, True) for s in image.shape]
        self.fft_image = self.fft_calc.rfftnd(image, self.fft_shape)

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
        """Calculate the minimum distance index. If multiple indices
        have the same minimum distance, choose one randomly.

        Args:
            template (numpy.ndarray): The template.
            coord_on_template (tuple): The coordinates of the pixel on the
                template.
        """
        distance_map = self.calc_distance_map(template)

        threshold = max(distance_map.min(), self.min_distance)
        inds = np.flatnonzero(distance_map <= threshold)
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
        matching_pixels = self.fft_calc.distance_map(self.fft_image, template,
                                                     self.fft_shape)
        slices = [slice(0, s_tr - s_te + 1)
                  for s_tr, s_te in zip(self.training_image.shape,
                                        template.shape)]
        matching_pixels = matching_pixels[tuple(slices)]
        # using direct slicing could slightly improve the performance
        # i_max = self.training_image.shape[0] - template.shape[0] + 1
        # j_max = self.training_image.shape[1] - template.shape[1] + 1
        # matching_pixels = matching_pixels[:i_max, :j_max]

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
            return tf.signal.irfft2d(image, shape)
        if len(shape) == 3:
            return tf.signal.irfft3d(image, shape)
        raise ValueError("Only 2D and 3D images are supported.")

    def distance_map(self, fft_image, template, shape):
        fft_template = tf.math.conj(tf.signal.rfftnd(template, shape))
        corr = tf.math.real(self.irfftnd(fft_image * fft_template, shape))
        return corr.numpy()


class SPFFT:
    def __init__(self):
        pass

    def rfftnd(self, image, shape, workers=None):
        return spfft.rfftn(image, s=shape, workers=workers)

    def distance_map(self, fft_image, template, shape, workers=None):
        fft_template = spfft.rfftn(template, s=shape, workers=workers).conj()
        corr = spfft.irfftn(fft_image * fft_template, workers=workers).real
        return corr
