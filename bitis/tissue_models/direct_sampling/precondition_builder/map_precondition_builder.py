from bitis.tissue_models.direct_sampling.precondition_builder.precondition_builder import PreconditionBuilder


class MapPreconditionBuilder(PreconditionBuilder):
    def __init__(self, simulation_size, image):
        PreconditionBuilder.__init__(self)

        self.simulation_size = simulation_size
        self.image           = image

    def set_rule(self, func):
        pass

    def build(self):
        pass
