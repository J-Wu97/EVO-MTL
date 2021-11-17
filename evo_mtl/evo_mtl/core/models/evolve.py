import numpy as np
from core.models.population import Population
import copy
from tools.eval import Evaluate
from tools import utils


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
        
        self.m_prob = m_prob
        self.x_prob = x_prob
        self.population_size = population_size

    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size)
        # all the initialized population should be saved
        utils.save_populations(gen_no=-1, pops=self.pops)

    def evaluate_fitness(self, gen_no, evaluated_num):
        print("evaluate fintesss")
        evaluate = Evaluate(self.pops)
        evaluate.parse_population(gen_no, evaluated_num)
        # all the initialized population should be saved
        utils.save_populations(gen_no=gen_no, pops=self.pops)
       
        print(self.pops)

    def recombinate(self, gen_no, evaluated_num, pop_size):
        print("mutation and crossover...")
        offspring_list = []
        cross_muta_num = 0
        while cross_muta_num < int(pop_size/2):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            if p1 != p2:
                
                offset1, offset2 = self.crossover(p1, p2) 
                offset1.mutation()
                offset2.mutation()
                offspring_list.append(offset1)
                offspring_list.append(offset2)
                cross_muta_num+=1
        # for _ in range(int(pop_size / 2)):
        #     
        #     p1 = self.tournament_selection()
        #     p2 = self.tournament_selection()
        # 
        #     offset1, offset2 = self.crossover(p1, p2)
        #     # mutation
        #     offset1.mutation()
        #     offset2.mutation()
        #     offspring_list.append(offset1)
        #     offspring_list.append(offset2)
        offspring_pops = Population(0)
        offspring_pops.set_populations(offspring_list)
        utils.save_offspring(gen_no, offspring_pops)
       
        self.pops.pops.extend(offspring_pops.pops)

        # evaluate these individuals
        evaluate = Evaluate(self.pops)
        evaluate.parse_population(gen_no, evaluated_num)
      
        self.pops.pops[pop_size:2 * pop_size] = offspring_pops.pops 
        utils.save_populations(gen_no=gen_no, pops=self.pops)
        # utils.save_each_gen_population(gen_no=gen_no, pops=self.pops)

    def environmental_selection(self, gen_no):
        assert (self.pops.get_pop_size() == 2 * self.population_size)
        print('environmental selection...')
        elitsam = 0.2
        e_count = int(np.floor(self.population_size * elitsam / 2) * 2)
        indi_list = self.pops.pops
        indi_list.sort(key=lambda x: x.MIou, reverse=False)
        elistm_list = indi_list[0:e_count] 

        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)
        np.random.shuffle(left_list)

        selected_winner = []
        selected_num = 0
        while selected_num < (self.population_size-e_count):
            i1 = utils.randint(0, len(left_list))  #
            i2 = utils.randint(0, len(left_list))
            winner = self.selection(left_list[i1], left_list[i2])  
            if winner not in selected_winner:
                elistm_list.append(winner)
                selected_num += 1

        # for _ in range(self.population_size - e_count):
        #     i1 = utils.randint(0, len(left_list))#
        #     i2 = utils.randint(0, len(left_list))
        #     winner = self.selection(left_list[i1], left_list[i2])
        #     selected_winner.append(winner)
        #     elistm_list.append(winner)

        self.pops.set_populations(elistm_list)
        utils.save_populations(gen_no=gen_no, pops=self.pops)
        # save_each_gen_population(gen_no=gen_no, pops=self.pops)
        if gen_no != 2:
            np.random.shuffle(self.pops.pops)

    def crossover(self, p1, p2):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()

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

        return p1, p2




    def tournament_selection(self):
        ind1_id = utils.randint(0, self.pops.get_pop_size())
        ind2_id = utils.randint(0, self.pops.get_pop_size())
        while ind1_id == ind2_id:
            ind1_id = utils.randint(0, self.pops.get_pop_size())
            ind2_id = utils.randint(0, self.pops.get_pop_size())
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1, ind2)
        return winner

    
    def selection(self, ind1, ind2):
        # Slack Binary Tournament Selection
        if ind1.loss_test < ind2.loss_test:
            return ind1
        else:
            return ind2


