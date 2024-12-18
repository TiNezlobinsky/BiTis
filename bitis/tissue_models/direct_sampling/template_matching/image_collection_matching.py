import numpy as np
from .template_maching import TemplateMatching


class ImageCollectionMatching(TemplateMatching):
    def __init__(self, training_images, template_size=None,
                 template_shape=None, min_distance=0.1, max_fraction=1.0):
        """
        Initializes an ImageCollectionMatching object.

        Args:
            training_images (list): A list of training images.
            template_size (int, optional): The size of the template.
            template_shape (tuple, optional): The shape of the template.
            min_distance (float, optional): The minimum distance threshold for
                accepting a match. Defaults to 0.1.
            max_fraction (float, optional): The maximum fraction of the
                training collection to scan. Defaults to 0.7
        """
        super().__init__()
        self._training_collection = []
        self._training_images = training_images

        if template_size is not None:
            self.template_size = template_size

        if template_shape is not None:
            self.template_shape = template_shape

        self.training_images = training_images

        self.min_distance = min_distance
        self.max_fraction = max_fraction

    @property
    def template_size(self):
        if np.all(self.template_shape != self.template_shape[0]):
            raise ValueError("Template shape is not uniform.")

        return self.template_shape[0]

    @template_size.setter
    def template_size(self, value):
        self.template_shape = (value, ) * len(self._training_images[0].shape)

    @property
    def training_images(self):
        return self._training_images

    @training_images.setter
    def training_images(self, images):
        self._training_collection = []
        self._training_images = images
        for image in images:
            self.build_collection(image)
        
        self._training_collection = np.array(self._training_collection)
        print(self._training_collection.shape)

    def build_collection(self, image):
        """
        Builds the training collection.

        Args:
            image (numpy.ndarray): The image to build the collection from.
        """
        coords = np.argwhere(image != 0)
        # make padding
        mask = (coords >= np.array(self.template_shape) // 2).all(axis=1)
        mask &= (coords < (np.array(image.shape) -
                           np.array(self.template_shape) +
                           np.array(self.template_shape) // 2)).all(axis=1)
        coords = coords[mask]
        # build collection
        for coord in coords:
            slices = tuple(slice(c - self.template_shape[i] // 2,
                                 c + self.template_shape[i] -
                                 self.template_shape[i] // 2)
                           for i, c in enumerate(coord))

            tr_template = image[slices]
            if tr_template.shape != self.template_shape:
                raise ValueError("Template has wrong shape.")

            self._training_collection.append(tr_template)

    def run(self, template, *args, **kwargs):
        """
        Runs the template matching.

        Args:
            template (numpy.ndarray): The template to match.

        Returns:
            numpy.ndarray: The matched template.
        """
        if np.count_nonzero(template != 0) == 0:
            id = np.random.randint(len(self._training_collection))
            tr_template = self._training_collection[id]
            return tr_template[*[s // 2 for s in tr_template.shape]]

        random_ids = np.random.permutation(len(self._training_collection))
        random_ids = random_ids[:int(self.max_fraction *
                                     len(self._training_collection))]

        dists = self.calc_distances(template)
        dists = dists[random_ids]

        if np.all(dists > self.min_distance):
            id = random_ids[np.argmin(dists)]
            tr_template = self._training_collection[id]
            return tr_template[*[s // 2 for s in tr_template.shape]]

        id = random_ids[np.argmax(dists <= self.min_distance)]

        tr_template = self._training_collection[id]
        return tr_template[*[s // 2 for s in tr_template.shape]]

    def calc_distances(self, template):
        """
        Calculates the distance between two templates.

        Args:
            template (numpy.ndarray): The simulation template.

        Returns:
            float: The distance between the two templates.
        """
        axis = tuple(range(1, self._training_collection.ndim))
        matching_pixels = (self._training_collection == template[None, ...]
                           ).sum(axis=axis)

        total_pixels = np.count_nonzero(template != 0)

        return 1 - matching_pixels / total_pixels
