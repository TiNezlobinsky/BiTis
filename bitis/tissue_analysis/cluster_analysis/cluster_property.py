import numpy as np
import pandas as pd
from skimage import measure, morphology, segmentation

from bitis.tissue_analysis.cluster_assembler import ClusterAssembler


class ClusterProperty:
    def __init__(self):
        pass

    @staticmethod
    def calc_measure_props(texture, prop_names):
        cluster_labels = ClusterAssembler.assemble(texture)
        return measure.regionprops_table(cluster_labels, intensity_image=texture, properties=prop_names)
    
    @staticmethod
    def calc_complexity(texture):
        props = ClusterProperty.calc_measure_props(texture, ['perimeter', 'area'])
        props['complexity'] = (props['perimeter'] ** 2
                               / (4 * np.pi * props['area']))
        return props
    
    @staticmethod
    def calc_axis_ratio(texture):
        props = ClusterProperty.calc_measure_props(texture, ['major_axis_length', 'minor_axis_length'])
        props['major_axis_length'] = np.where(props['major_axis_length'] >= 1,
                                              0.5 * props['major_axis_length'],
                                              0.5)
        props['minor_axis_length'] = np.where(props['minor_axis_length'] >= 1,
                                              0.5 * props['minor_axis_length'],
                                              0.5)
        props['axis_ratio'] = (props['major_axis_length']
                               / props['minor_axis_length'])
        return props
    
