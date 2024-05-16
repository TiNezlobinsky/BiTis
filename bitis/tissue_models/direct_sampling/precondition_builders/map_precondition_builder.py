from bitis.tissue_models.direct_sampling.precondition_builders.precondition_builder import PreconditionBuilder


class MapPreconditionBuilder(PreconditionBuilder):
    def __init__(self, simulation_size, image):
        """
        Initialize the MapPreconditionBuilder.

        Args:
            simulation_size (int): The size of the simulation.
            image (Image): The image used for preconditioning.
        """
        PreconditionBuilder.__init__(self)

        self.simulation_size = simulation_size
        self.image = image

    def set_rule(self, func):
        """
        Set the rule for the precondition builder.

        Args:
            func (function): The rule function.
        """
        pass

    def build(self):
        """
        Build the precondition.

        Returns:
            Precondition: The built precondition.
        """
        pass
