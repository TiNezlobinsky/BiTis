from bitis.tissue_models.direct_sampling.precondition.precondition_builder import PreconditionBuilder


class TexturePreconditionBuilder(PreconditionBuilder):
    """
    A class for building texture preconditions for tissue models.

    Args:
        simulation_size (int): The size of the simulation.

    Attributes:
        simulation_size (int): The size of the simulation.

    Methods:
        add_rectangle: Adds a rectangle to the texture precondition.
        add_circle: Adds a circle to the texture precondition.
        add_ellipse: Adds an ellipse to the texture precondition.
        interactive_draw: Allows interactive drawing of the texture precondition.
        build: Builds the texture precondition.
    """

    def __init__(self, simulation_size):
        """
        Initializes a TexturePreconditionBuilder object.

        Args:
            simulation_size (int): The size of the simulation.
        """
        PreconditionBuilder.__init__(self)

        self.simulation_size = simulation_size

    def add_rectangle(self, cell_type, center, a, b):
        """
        Adds a rectangle to the texture precondition.

        Args:
            cell_type (str): The type of the cell.
            center (tuple): The center coordinates of the rectangle.
            a (float): The length of the rectangle along the x-axis.
            b (float): The length of the rectangle along the y-axis.
        """
        pass

    def add_circle(self, cell_type, center, r):
        """
        Adds a circle to the texture precondition.

        Args:
            cell_type (str): The type of the cell.
            center (tuple): The center coordinates of the circle.
            r (float): The radius of the circle.
        """
        pass

    def add_ellipse(self, cell_type, center, a, b, orientation):
        """
        Adds an ellipse to the texture precondition.

        Args:
            cell_type (str): The type of the cell.
            center (tuple): The center coordinates of the ellipse.
            a (float): The length of the major axis of the ellipse.
            b (float): The length of the minor axis of the ellipse.
            orientation (float): The orientation angle of the ellipse.
        """
        pass

    def interactive_draw(self):
        """
        Allows interactive drawing of the texture precondition.
        """
        pass

    def build(self):
        """
        Builds the texture precondition.
        """
        pass
