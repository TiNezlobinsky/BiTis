import numpy as np
from .template_builder import TemplateBuilder


class FixedTemplateBuilder(TemplateBuilder):
    """
    Class for building fixed templates.

    Attributes:
        simulation_image (numpy.ndarray): The simulation image.
        template_shape (tuple): The shape of the template.
    """

    def __init__(self, simulation_image, template_size=None,
                 template_shape=None):
        """
        Args:
            simulation_image (numpy.ndarray): The simulation image.
            template_size (int): The size of the template.
            template_shape (tuple): The shape of the template.
        """
        super().__init__(simulation_image, template_size, template_shape)

    def build(self, coord):
        """
        Builds a template around target pixel.

        Args:
            coord (tuple): The coordinates of the pixel.

        Returns:
            numpy.ndarray: The template.
            tuple: The coordinates of the pixel on the template.
        """
        slices = [slice(c - self.template_shape[i] // 2,
                        c - self.template_shape[i] // 2 +
                        self.template_shape[i])
                  for i, c in enumerate(coord)]

        template = self.simulation_image[tuple(slices)]

        if template.shape != self.template_shape:
            print(slices, self.simulation_image.shape)
            raise ValueError("Sim Template shape is not correct.")

        coord_on_template = [self.template_shape[i] // 2
                             for i in range(len(coord))]

        return template, coord_on_template
