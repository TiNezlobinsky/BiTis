import numpy as np
from .template_builder import TemplateBuilder


class AdaptiveTemplateBuilder(TemplateBuilder):
    """
    Class for building adaptive templates.

    Attributes:
        max_known_pixels (int): The maximum number of known pixels in the
            template.
        max_template_size (int): The maximum size of the template.
        min_template_size (int): The minimum size of the template.
    """
    def __init__(self, simulation_image, max_known_pixels, max_template_size,
                 min_template_size=3):
        """
        Args:
            simulation_image (numpy.ndarray): The simulation image.
            max_known_pixels (int): The maximum number of known pixels in the
                template.
            max_template_size (int): The maximum size of the template.
            min_template_size (int): The minimum size of the template.
        """
        super().__init__(simulation_image)
        self.max_known_pixels = max_known_pixels
        self.max_template_size = max_template_size
        self.min_template_size = min_template_size

    def build(self, coord):
        """
        Builds a template around target pixel.

        The size of the template depends on the number of known pixels in the
        template. The size of the template is decreased until the number of
        known pixels is less than the ``max_known_pixels`` or the size of the
        template is less than ``min_template_size``.

        Args:
            coord (tuple): The coordinates of the pixel.

        Returns:
            numpy.ndarray: The template.
            tuple: The coordinates of the pixel on the template.
        """
        self.template_size = self.max_template_size
        template, coord_on_template = self._build(coord)
        if np.count_nonzero(template) <= self.max_known_pixels:
            return template, coord_on_template

        min_size, max_size = self.min_template_size, self.max_template_size
        while max_size - min_size > 1:
            mid_size = (min_size + max_size) // 2
            self.template_size = mid_size
            template, coord_on_template = self._build(coord)
            if np.count_nonzero(template) <= self.max_known_pixels:
                min_size = mid_size
            else:
                max_size = mid_size

        self.template_size = mid_size
        return self._build(coord)
        # for d_i in range(self.max_template_size, self.min_template_size, -1):
        #     self.template_size = d_i
        #     template, coord_on_template = self._build(coord)
        #     if np.count_nonzero(template) <= self.max_known_pixels:
        #         return template, coord_on_template

        # self.template_size = self.min_template_size

        # return self._build(coord)

    def _build(self, coord):
        """
        Builds a default template for the pixel.

        Args:
            coord (tuple): The coordinates of the pixel.

        Returns:
            numpy.ndarray: The template.
            tuple: The coordinates of the pixel on the template.
        """

        slices = []
        coord_on_template = []
        for i, t_size, s_size in zip(coord,
                                     self.template_shape,
                                     self.simulation_image.shape):
            coord_on_template.append(min(i, t_size // 2))
            slices.append(slice(max(0, i - t_size // 2),
                                min(s_size, i + t_size - t_size // 2)))

        template = self.simulation_image[tuple(slices)]

        return template, coord_on_template
