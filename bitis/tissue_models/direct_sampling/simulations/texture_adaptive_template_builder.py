import numpy as np
from scipy import spatial


class TextureAdaptiveTemplateBuilder:
    """
    Attributes:
        image (numpy.ndarray): The simulation image.
        max_size (int): The maximum number exiting neighbours in template.
        max_distance (int): The maximum distance for neighbours search.
            If number of neighbours is less than max_size,
            the default template with shape of 2 * max_distance is used.
    """

    def __init__(self, image, max_size=20, max_distance=30, min_distance=3):
        self.image = image
        self.max_size = max_size
        self.min_distance = min_distance
        self.max_distance = max_distance

    def build_defaut_template(self, image, i, j, max_distance):
        """
        Builds a default template for the pixel.

        Args:
            image (numpy.ndarray): The simulation image.
            i (int): The x coordinate of the pixel.
            j (int): The y coordinate of the pixel.
            max_distance (int): The maximum distance for neighbours search.
        """
        max_distance = int(2 * max_distance)
        i_min = i - max_distance // 2
        i_min = max(0, i_min)

        i_max = i + max_distance - max_distance // 2
        i_max = min(image.shape[0], i_max)

        j_min = j - max_distance // 2
        j_min = max(0, j_min)

        j_max = j + max_distance - max_distance // 2
        j_max = min(image.shape[1], j_max)
        return image[i_min: i_max, j_min: j_max]

    def build(self, i, j):
        """
        Builds a template for the pixel. The template contains at least
        max_size of simulated pixels or default template if there are no
        enough pixels. The template size decreases as more pixels are
        added to the simulation.

        Args:
            i (int): The x coordinate of the pixel.
            j (int): The y coordinate of the pixel.
        """
        coord = np.array([i, j])
        coords = np.argwhere(self.image > 0)

        if len(coords) < self.max_size:
            return self.build_defaut_template(self.image, *coord,
                                              self.max_distance)
        #  Find the nearest max_size already labeled pixels
        tree = spatial.KDTree(coords)
        d, ind = tree.query(coord, k=self.max_size,
                            distance_upper_bound=self.max_distance)
        ind = ind[d < np.inf]

        if len(ind) < self.max_size:
            return self.build_defaut_template(self.image, *coord,
                                              self.max_distance)

        coords = coords[ind]
        i_min, j_min = coords.min(axis=0)
        i_max, j_max = coords.max(axis=0)

        min_distance = int(2 * self.min_distance)
        if max(i_max - i_min, j_max - j_min) <= min_distance:
            return self.build_defaut_template(self.image, *coord,
                                              self.min_distance)
        return self.image[i_min:i_max, j_min:j_max]

    def update_image(self, i, j, value):
        """
        Updates the image with the given value at the given coordinates.

        Args:
            i (int): The x coordinate of the pixel.
            j (int): The y coordinate of the pixel.
            value (int): The value to set.
        """
        self.image[i, j] = value
