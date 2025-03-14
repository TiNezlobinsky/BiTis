import numpy as np
from scipy import fft as spfft
from .binary_image_matching import BinaryImageMatching


class ContinuousVariableMatching(BinaryImageMatching):
    def __init__(self,
                 training_image,
                 num_of_candidates=1,
                 min_known_pixels=1
                 ):
        super().__init__(training_image, num_of_candidates, min_known_pixels)

    @property
    def training_image(self):
        return self._original_image

    @training_image.setter
    def training_image(self, image):
        self._original_image = image
        image = image.copy().astype(np.float32)
        self.fft_shape = [spfft.next_fast_len(s, True) for s in image.shape]
        self.fft_image = self.fft_calc.rfftnd(image, self.fft_shape)
        self.fft_image2 = self.fft_calc.rfftnd(image ** 2, self.fft_shape)

    def compute_distance_map(self, template):
        """Compute the distance map between the template and the training
        image as the square root of the sum of the squared differences.

        Args:
            template (numpy.ndarray): The template.

        Returns:
            numpy.ndarray: The distance map.
        """
        template = template.astype(np.float32)

        fft_template = self.fft_calc.rfftnd(template, self.fft_shape)
        fft_ones = self.fft_calc.rfftnd((template != 0).astype(np.float32),
                                        self.fft_shape)
        fft_dist = (self.fft_calc.multiply(self.fft_image2, fft_ones) -
                    2 * self.fft_calc.multiply(self.fft_image, fft_template))
        dist = self.fft_calc.irfftnd(fft_dist, self.fft_shape)
        slices = [slice(0, s_tr - s_te + 1)
                  for s_tr, s_te in zip(self.training_image.shape,
                                        template.shape)]
        dist = dist[tuple(slices)]
        return np.sqrt(np.abs(dist + np.sum(template ** 2)))
