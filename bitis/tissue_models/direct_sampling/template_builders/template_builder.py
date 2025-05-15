from abc import ABC, abstractmethod
import numpy as np


class TemplateBuilder(ABC):
    """
    Base class for template builders.

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
        self.simulation_image = simulation_image

        if template_size is not None:
            self.template_size = template_size

        if template_shape is not None:
            self.template_shape = template_shape

    @property
    def template_size(self):
        if np.all(self.template_shape != self.template_shape[0]):
            raise ValueError("Template shape is not uniform.")

        return self.template_shape[0]

    @template_size.setter
    def template_size(self, value):
        self.template_shape = (value, ) * len(self.simulation_image.shape)

    @abstractmethod
    def build(self, coord):
        pass
