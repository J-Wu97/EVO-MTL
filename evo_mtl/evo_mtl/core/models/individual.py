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
    initialize a simple CNN network including one convolutional layer, one pooling layer, and one full connection layer
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
                    if np.random.random(1)>0.3:
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


if __name__ =='__main__':

    ind1 = Individual([2,2,3,3,3])
    ind1.initialize()
    ind2 = Individual([2,2,3,3,3])
    ind2.initialize()
    # print(ind1.indi_net1)
    # print(ind2.indi_net1)
    # print(ind1.indi_net2)
    # print(ind2.indi_net2)
    # ind1, ind2 = crossover(ind1,ind2)
    # print("-----------------------------------")
    # print(ind1.indi_net1)
    # print(ind2.indi_net1)
    # print(ind1.indi_net2)
    # print(ind2.indi_net2)

    # ind1 = copy.deepcopy(ind)
    # ind1_net1 = copy.deepcopy(ind1.indi_net1)
    # ind1_net2 = copy.deepcopy(ind1.indi_net2)
    # ind.mutation()
    # ind2 = copy.deepcopy(ind)
    # ind2_net1 = copy.deepcopy(ind2.indi_net1)
    # ind2_net2 = copy.deepcopy(ind2.indi_net2)
    #
    # print(ind1_net1,'\n', ind2_net1)
    # print(ind1_net2,'\n', ind2_net2)
    # print(is_mutation(ind1 , ind2))

    # x = []
    # y = []
    # print(ind.indi_net1)
    # print(ind.indi_net2)
    # for i in range(13):
    #     x.append(copy.deepcopy(ind.get_net1_layer_at(i)))
    #     y.append(copy.deepcopy(ind.get_net2_layer_at(i)))
    # for i in range(13):
    #     ind.indi_net1[i] = y[i]
    #     ind.indi_net2[i] = x[i]
    # for i in range(13):
    #     x[i] = y[i]
    # print(ind.indi_net1)
    # print(ind.indi_net2)
    # z = copy.deepcopy(ind.indi_net1)
    # w = copy.deepcopy(ind.indi_net2)
    # print(z)
    # print(w)

    # x1 = ind.get_net1_layer_at(0)
    # x2 = ind.get_net1_layer_at(12)
    # print(x1)
    # print(x2)
    # ind.indi_net1[0] = x2
    # print(ind.get_net1_layer_at(0))
    # print(ind.indi_net1)
