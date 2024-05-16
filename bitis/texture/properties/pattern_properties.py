
class PatternProperties:
    """
    Class representing the properties of a pattern.
    """

    def __init__(self):
        """
        Initializes a new instance of the PatternProperties class.
        """
        self.density     = float
        self.elongation  = float
        self.orientation = float
        self.compactness = float

    def __repr__(self):
        """
        Returns a string representation of the PatternProperties object.
        """
        return f"PatternProperties(density={self.density}, elongation={self.elongation}, orientation={self.orientation}, compactness={self.compactness})"


