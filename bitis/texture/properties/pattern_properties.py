
class PatternProperties:
    def __init__(self):
        self.density     = float
        self.elongation  = float
        self.orientation = float
        self.compactness = float

    def __repr__(self):
        return f"PatternProperties(density={self.density}, elongation={self.elongation}, orientation={self.orientation}, compactness={self.compactness})"


