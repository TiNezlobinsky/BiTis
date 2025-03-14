import numpy as np
from .binary_image_matching import BinaryImageMatching


class MultivariateVariableMatching(BinaryImageMatching):
    def __init__(self,
                 training_image,
                 joint_training_image,
                 joint_simulated_image,
                 n_candidates=1,
                 min_known_pixels=1):
        super().__init__(training_image, n_candidates, min_known_pixels)
        self.joint_training_image = joint_training_image
        self.joint_simulated_image = joint_simulated_image

    def run(self, coord, template_args):
        """Calculate the minimum distance index and return the corresponding
        pixel value.

        Args:
            template (numpy.ndarray): The template.
            coord_on_template (tuple): The coordinates of the target pixel on
                the template.

        Returns:
            int: The pixel value.
        """
        template, coord_on_template = template_args
        if self.n_candidates < 1:
            raise ValueError("Number of candidates must be greater than 0.")

        if np.count_nonzero(template) < self.min_known_pixels:
            return self.random_pixel()

        distance_map = self.compute_distance_map(coord, template)
        best_coord = self.find_best_match(distance_map, coord_on_template)
        res = (self.training_image[*best_coord],
               self.joint_training_image[*best_coord])
        return res

    def random_pixel(self):
        """Return a random pixel value from the training image.

        Returns:
            int: The pixel value.
        """
        i = np.random.randint(0, self.training_image.shape[0])
        j = np.random.randint(0, self.training_image.shape[1])
        return (self.training_image[i, j],
                self.joint_training_image[i, j])

    def compute_distance_map(self, coord, template):
        """Compute the distance map between the template and the training
        image as the square root of the sum of the squared differences and
        the absolute difference between the joint training image and the joint
        simulated image.

        Args:
            coord (tuple): The coordinates of the target pixel on the template.
            template (numpy.ndarray): The template.

        Returns:
            numpy.ndarray: The distance map.
        """
        distance_map = super().compute_distance_map(template)
        joint_map = np.abs(self.joint_training_image -
                           self.joint_simulated_image[*coord])
        i_min = template.shape[0] // 2
        j_min = template.shape[1] // 2
        i_max = i_min + distance_map.shape[0]
        j_max = j_min + distance_map.shape[1]
        distance_map = distance_map + joint_map[i_min:i_max, j_min:j_max]
        return distance_map
