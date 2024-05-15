import numpy as np

from skimage import measure, morphology, segmentation


class ObjectAnalysis:
    def __init__(self):
        pass

    def calc_density(self, tex):
        return tex[tex == 2].size/tex.size

    def build_props(self, tex, area_threshold=5, connectivity=1, clear_border=False):
        mask = morphology.remove_small_objects(tex > 1, area_threshold)
        if clear_border:
            mask = segmentation.clear_border(mask)
        labels = measure.label(mask, background=False, connectivity=connectivity)
        props = measure.regionprops_table(labels,
                                          properties=('label',
                                                      'area',
                                                      'centroid',
                                                      'major_axis_length',
                                                      'minor_axis_length',
                                                      'orientation'))
        
        area_quant = np.quantile(props['area'], 0.75)
        quant_props = {"area": [], "major_axis_length": [], "minor_axis_length": [], "orientation": [], "label": []}
        for i in range(len(props['label'])):
            if props['area'][i] >= area_quant:
                quant_props['area'].append(props['area'][i])
                quant_props['major_axis_length'].append(props['major_axis_length'][i])
                quant_props['minor_axis_length'].append(props['minor_axis_length'][i])
                quant_props['orientation'].append(props['orientation'][i])
                quant_props['label'].append(props['label'][i])

        return props
    
    def calc_anisotropy(self, props):
        major_axis = np.mean(props['major_axis_length'])
        minor_axis = np.mean(props['minor_axis_length'])

        return major_axis/minor_axis, np.mean(props['orientation'])
    
    def calc_compactness(self, tex, props):
        area_array = props['area']
        quant = np.quantile(area_array, 0.75)
        return np.sum(area_array[area_array > quant])/tex.size
    
    def get_pattern(self, tex, area_threshold=5, connectivity=1, clear_border=False):
        props = self.build_props(tex, area_threshold, connectivity, clear_border)

        density = self.calc_density(tex)
        elongation, orientation = self.calc_anisotropy(props)
        compactness = self.calc_compactness(tex, props)

        return density, elongation, orientation, compactness


    


