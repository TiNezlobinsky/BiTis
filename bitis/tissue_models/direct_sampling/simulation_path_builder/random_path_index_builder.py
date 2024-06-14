import numpy as np


class RandomPathIndexBuilder:
    """
    Attributes:
        out_shape (tuple): The size of the output.
    """
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def build(self):
        """
        Builds a random path for simulation based on the given output shape.

        Returns:
            numpy.ndarray: A random permutation of coordinates for
                the simulation path.
        """
        X, Y = np.meshgrid(np.arange(self.out_shape[0]),
                           np.arange(self.out_shape[1]))
        coords = np.array([X.ravel(), Y.ravel()]).T
        coords = np.random.permutation(coords)
        return coords
