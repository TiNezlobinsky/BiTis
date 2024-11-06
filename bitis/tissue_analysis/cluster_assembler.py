import numpy as np
import pandas as pd
from skimage import measure, segmentation


class ClusterAssembler:

    @staticmethod
    def assemble(texture, connectivity=1, clear_border=False):
        mask = texture > 0

        if clear_border:
            mask = segmentation.clear_border(mask)

        cluster_labels = measure.label(mask, background=False,
                               connectivity=connectivity)

        return cluster_labels