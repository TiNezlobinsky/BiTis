import numpy as np


class AdaptiveTrainingDataBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build(i, j, tex_shape, tr_shape):
        ind = np.array([i, j])
        tr_shape = np.array(tr_shape)
        tex_shape = np.array(tex_shape)

        ind_min = (ind - tr_shape // 2)
        ind_min[ind_min < 0] = 0
        ind_max = ind_min + tr_shape

        ind_max[ind_max > tex_shape] = tex_shape[ind_max > tex_shape]
        ind_min = ind_max - tr_shape

        i_min, j_min = ind_min
        i_max, j_max = ind_max

        if i_max - i_min != tr_shape[0]:
            print('Error: i_max - i_min != tr_shape[0]')
        if j_max - j_min != tr_shape[1]:
            print('Error: j_max - j_min != tr_shape[1]')

        return i_min, i_max, j_min, j_max
