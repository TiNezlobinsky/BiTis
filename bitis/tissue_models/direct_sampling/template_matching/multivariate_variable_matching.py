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
        training_pixel = self.training_image[*best_coord]
        joint_training_pixel = self.joint_training_image[*best_coord]

        if self.joint_training_image.ndim > self.training_image.ndim:
            joint_training_pixel = joint_training_pixel.mean()

        return (training_pixel, joint_training_pixel)

    def random_pixel(self):
        """Return a random pixel value from the training image.

        Returns:
            int: The pixel value.
        """
        coord = [np.random.randint(0, ts)
                 for ts in self.training_image.shape]
        training_pixel = self.training_image[*coord]
        joint_training_pixel = self.joint_training_image[*coord]

        if self.joint_training_image.ndim > training_pixel.ndim:
            joint_training_pixel = joint_training_pixel.mean()

        return (training_pixel, joint_training_pixel)

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

        if joint_map.ndim > distance_map.ndim:
            joint_map = joint_map.mean(axis=-1)

        slices = tuple([slice(ts // 2, ts // 2 + ds)
                        for ts, ds in zip(template.shape, distance_map.shape)])

        distance_map = distance_map + joint_map[slices]
        return distance_map
