import os
import numpy as np
# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >temp')
#
# memory_gpu=[int(x.split()[2]) for x in open('temp','r').readlines()]
# print("动态选择的显卡是",str(np.argmax(memory_gpu)))
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
# os.system('rm temp')


senvecs_1 = np.array([[1, 2, 3],
                      [1, 2, 3]])
senvecs_2 = np.array([[4, 5, 6],
                      [4, 5, 6]])
scores = np.sqrt(np.sum(np.square(senvecs_1 - senvecs_2), axis=1))
print(type(scores))