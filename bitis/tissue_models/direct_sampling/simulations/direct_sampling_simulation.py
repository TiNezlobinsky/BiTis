

class DirectSamplingSimulation:
    def __init__(self, input_image, training_image, path_builder, distance_builder, template_size=15):
        self.input_image      = input_image
        self.training_image   = training_image
        self.path_builder     = path_builder
        self.distance_builder = distance_builder
        self.output_image     = None

        self.template_size = template_size

    def build_template(self, i, j):
        # Could be a separate class
        """
        Builds a default template for the pixel.

        Args:
            i (int): The x coordinate of the pixel.
            j (int): The y coordinate of the pixel.
        """
        i_min = i - self.template_size // 2
        i_min = max(0, i_min)

        i_max = i + self.template_size - self.template_size // 2
        i_max = min(self.input_image.shape[0], i_max)

        j_min = j - self.template_size // 2
        j_min = max(0, j_min)

        j_max = j + self.template_size - self.template_size // 2
        j_max = min(self.input_image.shape[1], j_max)

        return self.input_image[i_min: i_max, j_min: j_max]

    def run(self):
        coords = self.path_builder.build()
        for coord in coords:
            i, j = coord
            template = self.build_template(*coord)
            closest_pixel = self.distance_builder.build(template, i, j)
            self.input_image[i, j] = closest_pixel

        return self.input_image

