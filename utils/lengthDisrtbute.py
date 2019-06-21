lens = {}
with open("../test.txt",mode="r",encoding="utf-8") as fr:
    for line in fr:
        line = line.strip()
        if line != "":
            temp = [len(x) for x in line.split("---cvtenlp---")]
            for k in temp:
                if k in lens:
                    lens[k]+=1
                else:
                    lens[k]=1

lens_list = sorted(lens.items(),key=lambda item:item[0])
xs=[]
ys=[]
for x,y in lens_list:
    xs.append(x)
    ys.append(y)

import matplotlib.pyplot as plt

num_list = [1.5, 0.6, 7.8, 6]
plt.bar(xs, ys)


plt.show()
