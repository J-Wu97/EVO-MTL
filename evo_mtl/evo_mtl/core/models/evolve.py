import numpy as np
from core.models.population import Population
import copy
from tools.eval import Evaluate
from tools import utils
'''
进化过程：包括初始化种群，适应度评价，交叉变异，环境选择等
1.所以引入种群
2.引入评估->即评估其CNN架构适应度
3.对个体进行交叉变异
4.筛选
'''

class Evolve_CNN:
    def __init__(self, m_prob, x_prob, population_size):
        '''
        :param m_prob:
        :param m_eta:
        :param x_prob:
        :param x_eta:
        :param population_size:
        :param batch_size:
        '''
        #进化过程中，所需要的基本的交叉、变异率等传进来
        self.m_prob = m_prob
        self.x_prob = x_prob
        self.population_size = population_size

    def initialize_popualtion(self):#初始化种群
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size)
        # all the initialized population should be saved
        utils.save_populations(gen_no=-1, pops=self.pops)

    def evaluate_fitness(self, gen_no, evaluated_num):
        print("evaluate fintesss")
        evaluate = Evaluate(self.pops)#评估种群
        evaluate.parse_population(gen_no, evaluated_num)
        # all the initialized population should be saved
        utils.save_populations(gen_no=gen_no, pops=self.pops)
        # '''
        # 这两个地方为什么还要存储
        # '''
        # utils.save_each_gen_population(gen_no=gen_no, pops=self.pops)
        print(self.pops)

    def recombinate(self, gen_no, evaluated_num, pop_size):
        print("mutation and crossover...")
        offspring_list = []
        cross_muta_num = 0
        while cross_muta_num < int(pop_size/2):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            if p1 != p2:
                # crossover产生新个体
                offset1, offset2 = self.crossover(p1, p2)  # 先进行交叉 产生的后代进行变异
                # mutation对新个体进行变异，每个个体的变异是自己内部的事情，所以我们把功能的实现放在了个体内部
                offset1.mutation()
                offset2.mutation()
                offspring_list.append(offset1)
                offspring_list.append(offset2)
                cross_muta_num+=1
        # for _ in range(int(pop_size / 2)):
        #     #选择优秀的父母
        #     p1 = self.tournament_selection()
        #     p2 = self.tournament_selection()
        #     # crossover产生新个体
        #     offset1, offset2 = self.crossover(p1, p2)#先进行交叉 产生的后代进行变异
        #     # mutation对新个体进行变异，每个个体的变异是自己内部的事情，所以我们把功能的实现放在了个体内部
        #     offset1.mutation()
        #     offset2.mutation()
        #     offspring_list.append(offset1)
        #     offspring_list.append(offset2)
        offspring_pops = Population(0)#实例化种群 并初始化
        offspring_pops.set_populations(offspring_list)#新的种群
        utils.save_offspring(gen_no, offspring_pops)
        '''
        offspring_pops就是一个新的种群了。
        '''
        self.pops.pops.extend(offspring_pops.pops)#新的老的个体做一个汇总，当然新的都是经过交叉变异的

        # evaluate these individuals
        evaluate = Evaluate(self.pops)#再评估
        evaluate.parse_population(gen_no, evaluated_num)
        '''
        上面已经扩展了，为啥还要扩展？
        '''
        self.pops.pops[pop_size:2 * pop_size] = offspring_pops.pops #两倍长
        utils.save_populations(gen_no=gen_no, pops=self.pops)
        # utils.save_each_gen_population(gen_no=gen_no, pops=self.pops)

    def environmental_selection(self, gen_no):#环境选择
        assert (self.pops.get_pop_size() == 2 * self.population_size)#检查是否合法
        print('environmental selection...')
        elitsam = 0.2
        e_count = int(np.floor(self.population_size * elitsam / 2) * 2)
        indi_list = self.pops.pops
        indi_list.sort(key=lambda x: x.MIou, reverse=False)
        # 这里要升序排序才可以，mean_loss越小越好，即reverse=Flase，就根据meanloss排序
        elistm_list = indi_list[0:e_count]  #根据我们的精英率选择精英

        left_list = indi_list[e_count:]#挑选完精英之后留下的人
        np.random.shuffle(left_list)#对留下的人进行随机打乱
        np.random.shuffle(left_list)

        selected_winner = []
        selected_num = 0
        while selected_num < (self.population_size-e_count):
            i1 = utils.randint(0, len(left_list))  #
            i2 = utils.randint(0, len(left_list))
            winner = self.selection(left_list[i1], left_list[i2])  # 对剩下的表现的不好的列表在进行筛选
            if winner not in selected_winner:
                elistm_list.append(winner)
                selected_num += 1

        # for _ in range(self.population_size - e_count):
        #     i1 = utils.randint(0, len(left_list))#
        #     i2 = utils.randint(0, len(left_list))
        #     winner = self.selection(left_list[i1], left_list[i2])#对剩下的表现的不好的列表在进行筛选
        #     selected_winner.append(winner)
        #     elistm_list.append(winner)

        self.pops.set_populations(elistm_list)#替代原有的种群了
        utils.save_populations(gen_no=gen_no, pops=self.pops)
        # save_each_gen_population(gen_no=gen_no, pops=self.pops)
        if gen_no != 2:
            #最后一代不用shuffle
            np.random.shuffle(self.pops.pops)#再随机打乱种群

    def crossover(self, p1, p2):#交叉肯定是个体的事情
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()
        '''
        将两种不同的层次基因，交叉是针对的基因。
        分开然后同类型的之间对其交叉 unit就是基因
        '''

        l = p1.get_indi_size()

        for i in range(l):
            unit_p1_net1 = copy.deepcopy(p1.get_net1_layer_at(i))
            unit_p1_net2 = copy.deepcopy(p1.get_net2_layer_at(i))
            unit_p2_net1 = copy.deepcopy(p2.get_net1_layer_at(i))
            unit_p2_net2 = copy.deepcopy(p2.get_net2_layer_at(i))
            if utils.flip(self.x_prob):
                p1.indi_net1[i] = unit_p2_net1
                p2.indi_net1[i] = unit_p1_net1
            if utils.flip(self.x_prob):
                p1.indi_net2[i] = unit_p2_net2
                p2.indi_net2[i] = unit_p1_net2

        return p1, p2#不会对原来的数据产生影响 因为是deepcopy




    def tournament_selection(self):#二元锦标赛选择法， 从种群中随机选择两个个体，竞争，得到优胜者
        ind1_id = utils.randint(0, self.pops.get_pop_size())
        ind2_id = utils.randint(0, self.pops.get_pop_size())
        while ind1_id == ind2_id:
            ind1_id = utils.randint(0, self.pops.get_pop_size())
            ind2_id = utils.randint(0, self.pops.get_pop_size())
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1, ind2)
        return winner

    '''
    所以说这里的选择不仅仅考虑了准确度，同时还考虑了模型的复杂度（参数个数）
    '''
    def selection(self, ind1, ind2):
        # Slack Binary Tournament Selection
        MIou = 0.0120
        Mean = 0.0120
        # 一个四层的cnn（fliter_size均为3的话，param个数为220万），所以决定param用比例,mean用绝对的数值
        if ind1.MIou > ind2.MIou:
            # 此时ind2性能比1好
            if ind1.MIou - ind2.MIou > MIou:  # 差值越大说明1的性能越差
                return ind1 #差值超过了一定的范围，那么直接按照准确率选择 否则 我们看复杂度
            else:
                # 在没有差到超过阈值mean_threshold的情况下，如果2的复杂度相对1的百分比没有超过阈值则返回2，反之则返回1
                # 因为老是有零除错误，所以改为乘号
                if ind1.Mean - ind2.Mean > Mean:
                    return ind2
                else:
                    return ind1
        else:
            # 此时ind1性能比2好
            if ind2.MIou - ind1.MIou > MIou:
                return ind2
            else:
                if ind2.Mean - ind1.Mean > Mean:
                    return ind1
                else:
                    return ind2


# if __name__ == '__main__':
#
#     # mutation 测试
#     cnn = Evolve_CNN(0.3, 0.7, 4)
#     cnn.initialize_popualtion()
#     for i in range(4):
#         print(cnn.pops.get_individual_at(i).indi_net1)
#     cnn.recombinate(1, 0, 4)
#     for i in range(cnn.pops.get_pop_size()):
#         print(cnn.pops.get_individual_at(i).indi_net1)
    # if ind == cnn.pops.pops[0]:
    #     print('不在里面哦')
    # else:
    #     print('在里面哦')
    # print(cnn.pops)
    # ind1 = cnn.tournament_selection()
    # ind2 = cnn.tournament_selection()
    # print('p1->', '\n', ind1.indi_net1, '\n', ind1.indi_net2)
    # print('p2->', '\n', ind2.indi_net1, '\n', ind2.indi_net2)
    # new_p1, new_p2 = cnn.crossover(ind1, ind2)
    # print('new_p1->', '\n', new_p1.indi_net1, '\n', new_p1.indi_net2)
    # print('new_p2->', '\n', new_p2.indi_net1, '\n', new_p2.indi_net2)
    # new_p1.mutation()
    # new_p2.mutation()
    # print('nnp1->','\n',new_p1.indi_net1,'\n', new_p1.indi_net2)
    # print('nnp2->', '\n', new_p2.indi_net1, '\n', new_p2.indi_net2)


    # # crossover测试
    # cnn = Evolve_CNN(0.2, 0.9, 4)
    # indi1 = Individual([2,2,3,3,3])
    # indi2 = Individual([2,2,3,3,3])
    #
    # indi1.initialize()
    # indi2.initialize()
    # print(indi1.get_indi_size())
    # for i in range(indi1.get_indi_size()):
    #     cur_unit = indi1.get_layer_at(i)
    #     print(cur_unit)
    # print('------------------------')
    # print(indi2.get_layer_size())
    # for i in range(indi2.get_layer_size()):
    #     cur_unit = indi2.get_layer_at(i)
    #     print(cur_unit)
    # print('------------------------')
    #
    # print('crossover---------------')
    # indi1, indi2 = cnn.crossover(indi1, indi2)
    # print(indi1.get_layer_size())
    # for i in range(indi1.get_layer_size()):
    #     cur_unit = indi1.get_layer_at(i)
    #     print(cur_unit)
    # print('------------------------')
    # print(indi2.get_layer_size())
    # for i in range(indi2.get_layer_size()):
    #     cur_unit = indi2.get_layer_at(i)
    #     print(cur_unit)
    # print('------------------------')
