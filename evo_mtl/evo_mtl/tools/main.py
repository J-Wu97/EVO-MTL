import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)
lib_path1 = r'/home/jiewu/ga-mtl'
sys.path.append(lib_path1)
lib_path2 = r'/home/jiewu/ga-mtl/core'
print(lib_path2)
sys.path.append(lib_path2)
lib_path3 = r'/home/jiewu/ga-mtl/core/data'
print(lib_path3)
sys.path.append(lib_path3)
lib_path4 = r'/home/jiewu/ga-mtl/core/models'
print(lib_path4)
sys.path.append(lib_path4)

from core.models.evolve import Evolve_CNN
from tools.utils import *


def begin_evolve(m_prob, x_prob, pop_size, total_generation_num):
    # 只用于创建初始种群，保存为pops.dat
    cnn = Evolve_CNN(m_prob,  x_prob,  pop_size)
    cnn.initialize_popualtion()
    cnn.evaluate_fitness(0, 0)
    for cur_gen_no in range(total_generation_num):
        print('The {}/{} generation'.format(cur_gen_no+1, total_generation_num))
        cnn.recombinate(cur_gen_no+1,cnn.pops.get_pop_size(), pop_size)
        cnn.environmental_selection(cur_gen_no+1)


# def restart_evolve(m_prob, x_prob, pop_size, total_gene_number):
#     gen_no, pops, _ = load_population()
#     evaluated_num = pops.get_evaluated_pop_size()
#     cnn = Evolve_CNN(m_prob, x_prob, pop_size)
#     cnn.pops = pops
#     if gen_no < 0:
#         print('first to evaluate...')
#         # cnn.evaluate_fitness(1, )
#     else:
#         for cure_gen_no in range(gen_no+1, total_gene_number+1):
#             print('continue to evolve from the {}/{} generation...'.format(cure_gen_no, total_gene_number))
#             cnn.recombinate(cure_gen_no,)
#             cnn.environmental_selection(cure_gen_no)
#
#     if evaluated_num != pop_size * 2:  # 接着上一代没跑完的继续evaluate完，且不是第一代
#         print('continue to evaluate indi:{}...'.format(evaluated_num))
#         cnn.evaluate_fitness(gen_no, evaluated_num)
#     evaluated_num = pop_size

    # # 判断有没有经历environmental_selection
    # if pops.get_evaluated_pop_size() == pop_size * 2:
    #     cur_gen_no = gen_no
    #     cnn.environmental_selection(cur_gen_no)
    # for cur_gen_no in range(gen_no + 1, total_gene_number + 1):
    #     print('Continue to evolve from the {}/{} generation...'.format(cur_gen_no, total_gene_number))
    #     cnn.recombinate(cur_gen_no, evaluated_num, pop_size)
    #     evaluated_num = pop_size
    #     cnn.environmental_selection(cur_gen_no)


if __name__ == '__main__':
    # train_data, validation_data, test_data = get_mnist_data()

    total_generation_number = 10  # total generation number
    pop_size = 30

    # # 测试
    # gen_no, pops, create_time = load_population()
    # print(gen_no)
    # print(pops)
    # print(pops.get_evaluated_pop_size())


    begin_evolve(0.9, 0.2,pop_size, total_generation_number)
    # restart_evolve(0.9, 0.2, pop_size, total_generation_number)
