

class Texture:
    def __init__(self):
        self.matrix     = None
        self.properties = {}

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        self.matrix[index] = value

    def __repr__(self):
        return f"Texture({self.matrix})"