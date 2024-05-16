

class Texture:
    """
    A class representing a texture.

    Attributes:
    - matrix (list): The matrix representing the texture.
    - properties (dict): Additional properties of the texture.

    Methods:
    - __getitem__(index): Get the value at the specified index in the texture matrix.
    - __setitem__(index, value): Set the value at the specified index in the texture matrix.
    - __repr__(): Return a string representation of the Texture object.
    """

    def __init__(self):
        self.matrix     = None
        self.properties = {}

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        self.matrix[index] = value

    def __repr__(self):
        return f"Texture({self.matrix})"