from bitis.tissue_models.direct_sampling.precondition_builder.precondition_builder import PreconditionBuilder


class MapPreconditionBuilder(PreconditionBuilder):
    def __init__(self, tex_size, image):
        PreconditionBuilder.__init__(self)

        self.tex_size = tex_size
        self.image    = image

    def set_rule(self, func):
        pass

    def build(self):
        pass
