import numpy as np
from skimage import transform
from .foobar_matching import FooBarMatching
from .binary_image_matching import BinaryImageMatching


class RotatedImagesMatching(FooBarMatching):
    """
    Attributes:
        image (numpy.ndarray): The training image.
        distance_threshold (float): The distance threshold.
    """

    def __init__(self,
                 base_image,
                 angle_map,
                 base_angle=0,
                 **kwargs):

        super().__init__()
        self.angle_map = np.round(angle_map).astype(int)
        self.buid_template_matchers(base_image, base_angle)
        self._best_index = -1

    def buid_template_matchers(self, base_image, base_angle, **kwargs):
        angle_list = np.unique(self.angle_map)

        for angle in angle_list:
            rotated_image = self.rotate_image(base_image, angle - base_angle)
            template_matcher = BinaryImageMatching(rotated_image, **kwargs)
            self.template_matchers[angle] = template_matcher

    def run(self, template, coord, coord_on_template, **kwargs):
        selector = self.angle_map[*coord]
        self._best_index = -1
        return super().run(template,
                           selector,
                           coord_on_template=coord_on_template,
                           **kwargs)

    def rotate_image(self, base_image, angle):
        theta = np.radians(angle)

        w, h = base_image.shape[1], base_image.shape[0]
        cx, cy = w / 2, h / 2

        # Rotated center position
        cx_rot = cx * np.cos(theta) - cy * np.sin(theta)
        cy_rot = cx * np.sin(theta) + cy * np.cos(theta)

        # Translation needed to recenter
        tx = cx - cx_rot
        ty = cy - cy_rot

        tform = transform.EuclideanTransform(rotation=theta,
                                             translation=(tx, ty))
        tf_img = transform.warp(base_image, tform.inverse, preserve_range=True,
                                order=0)
        return tf_img
