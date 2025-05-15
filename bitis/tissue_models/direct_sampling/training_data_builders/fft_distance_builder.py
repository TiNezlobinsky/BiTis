import scipy.fft as spfft
import numpy as np
from scipy import signal

from ..template_matching.distance_map.distance_map_builder import DistanceBuilder


class FFTDistanceBuilder(DistanceBuilder):
    def __init__(self, image, distance_threshold=0.0):
        super().__init__(image, distance_threshold)
        self.image = image

    @property
    def image(self):
        image = self._image.copy()
        image[image == -1] = 2
        return image

    @image.setter
    def image(self, value):
        value = value.copy()
        value[value == 2] = -1
        self._image = value.astype(np.float32)
        # self.fft_shape = [spfft.next_fast_len(s, True) for s in value.shape]
        # self.fft_image = spfft.rfftn(value, s=self.fft_shape)

    def calc_distance_map(self, image, template):
        """Calculate the distance map between the training image and
        the template.

        Args:
            image (numpy.ndarray): The training image.
            template (numpy.ndarray): The template.
        """
        template = template.copy()
        template[template == 2] = -1
        mist_match = signal.correlate(self._image.astype(np.float32),
                                      template.astype(np.float32),
                                      mode='valid', method='fft')
        # template = template.copy()
        # template[template == 2] = -1
        # fft_template = spfft.rfftn(template, s=self.fft_shape)
        # mist_match = spfft.irfftn(self.fft_image * fft_template).real

        # i_max = self._image.shape[0] - template.shape[0] + 1
        # j_max = self._image.shape[1] - template.shape[1] + 1
        # mist_match = mist_match[:i_max, :j_max]

        mist_match = 0.5 * (mist_match + np.count_nonzero(template != 0))
        return mist_match / template.size
