from bitis.tissue_models.direct_sampling.precondition_builders.precondition_builder import PreconditionBuilder


class TexturePreconditionBuilder(PreconditionBuilder):
    def __init__(self, simulation_size):
        PreconditionBuilder.__init__(self)

        self.simulation_size = simulation_size

    def add_rectangle(self, cell_type, center, a, b):
        pass

    def add_circle(self, cell_type, center, r):
        pass

    def add_ellipse(self, cell_type, center, a, b, orientation):
        pass

    def interactive_draw(self):
        pass

    def build(self):
        pass
