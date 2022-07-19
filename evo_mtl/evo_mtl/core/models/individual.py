import numpy as np
import copy
from tools import utils
class Individual:

    def __init__(self, stage_config, x_prob=0.9,  m_prob=0.3):
        self.stage_config = stage_config
        self.x_prob = x_prob
        self.m_prob = m_prob
        #metric
        self.complxity = 0
        self.MIou = 0.
        self.PAcc = 0.
        self.Mean = 0.
        self.Median = 0.
        self.RMSE = 0.
        self.within_11 = 0.
        self.within_22 = 0.
        self.within_30 = 0.
        self.within_45 = 0.
        self.indi_net1 = None
        self.indi_net2 = None

    def clear_state_info(self):
        self.complxity = 0

    '''
    initialize a simple CNN network
    '''

    def initialize(self):
        self.indi_net1 = self.init_one_individual()
        self.indi_net2 = self.init_one_individual()

    def get_indi_size(self):
        return np.sum(self.stage_config)

    def init_one_individual(self):
        limit = 3
        network_depth = np.sum(self.stage_config)
        stage_depths = np.cumsum([0] + self.stage_config)
        matrix = np.zeros((network_depth, network_depth)).astype('int')
        for i in range(network_depth):
            j_limit = stage_depths[np.argmax(stage_depths > i) - 1]
            for j in range(network_depth):
                if j <= i and i - j < limit and j >= j_limit:
                    if np.random.random(1)>0.5:
                        matrix[i, j] = 1.
        return matrix


    def mutation(self):
        matrix1 = self.indi_net1
        matrix2 = self.indi_net2

        if np.random.random(1) < 1:
            limit = 3
            depth = matrix1.shape[0]
            stage_depths = np.cumsum([0] + self.stage_config)
            for i in range(depth):
                j_limit = stage_depths[np.argmax(stage_depths > i) - 1]
                for j in range(depth):
                    if j <= i and i - j < limit and j >= j_limit:
                        if matrix1[i, j] == 1:
                            if utils.flip(self.m_prob):
                                matrix1[i, j] = 0
                        else:
                            if utils.flip(self.m_prob):
                                matrix1[i, j] = 1
                        if matrix2[i, j] == 1:
                            if utils.flip(self.m_prob):
                                matrix2[i, j] = 0
                        else:
                            if utils.flip(self.m_prob):
                                matrix2[i, j] = 1
        self.indi_net1 = matrix1
        self.indi_net2 = matrix2

    def get_net1_layer_at(self, i):
        return self.indi_net1[i]

    def get_net2_layer_at(self, i):
        return self.indi_net2[i]



def is_mutation(indi1, indi2):
    if (indi1.indi_net1 == indi2.indi_net1).all():
        if (indi1.indi_net2 == indi2.indi_net2).all():
            return False
    else:
        return True

