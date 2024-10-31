import numpy as np


class RandomPathIndexBuilder:
    """
    Attributes:
        out_mask (numpy.ndarray): The output mask.
    """
    def __init__(self, out_shape=None, out_mask=None):

        if out_mask is None:
            out_mask = np.ones(out_shape)

        self.out_mask = out_mask

    def build(self):
        """
        Builds a random path for simulation based on the given output shape.

        Returns:
            numpy.ndarray: A random permutation of coordinates for
                the simulation path.
        """
        mask = np.zeros_likes(self.out_mask)
        while np.all(mask == self.out_mask):
            
        coords = np.argwhere(self.out_mask)
        coords = np.random.permutation(coords)
        return coords
