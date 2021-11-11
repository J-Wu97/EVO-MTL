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
from tools.utils import load_population

gen_no, pops, _ = load_population()
print('gen_no:{}'.format(gen_no))
for i in range(pops.get_pop_size()):
    print(pops.pops[i])
    print(pops.pops[i].MIou)
