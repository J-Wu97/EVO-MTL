from core.models.individual import Individual
import copy

class Population:

    def __init__(self, num_pops):
        self.num_pops = num_pops
        self.pops = []
        for i in range(num_pops):
            indi = Individual([2,2,3,3,3])
            indi.initialize()
            self.pops.append(indi)

    def get_individual_at(self, i):
        return self.pops[i]

    def get_pop_size(self):
        return len(self.pops)

    def set_populations(self, new_pops):
        self.pops = new_pops

    def get_evaluated_pop_size(self):
        evaluated_size = 0
        for i in range(self.get_pop_size()):
            indi = self.get_individual_at(i)
            if indi.MIou != 0:
                evaluated_size = evaluated_size + 1
        return evaluated_size





    def __str__(self):
        _str = []
        for i in range(self.get_pop_size()):
            _str.append(str(self.get_individual_at(i)))
        return '\n'.join(_str)
if __name__ == '__main__':
    pop = Population(4)
    len_pop = pop.get_pop_size()
    for i in range(len_pop):
        pop.get_individual_at(i).mutation()
        indi = pop.get_individual_at(i).indi
        print(indi)
    print(pop)