import numpy as np
from .single_image_matching import SingleImageMatching


class MultivariateVariableMatching(SingleImageMatching):
    def __init__(self, training_image, joint_image, num_of_candidates=1,
                 min_known_pixels=1, use_tf=False):
        super().__init__(training_image, num_of_candidates, min_known_pixels,
                         use_tf)
        self.joint_image = joint_image

    def compute_distance_map(self, template, coord_on_template):
        distance_map = super().compute_distance_map(template, coord_on_template)
        joint_map = np.abs(self.joint_image - template[*coord_on_template])
        distance_map = distance_map + joint_map
        return distance_map
