import os
from os import listdir
import os.path
import numpy as np
import random

img_dir= "data/bin-images/"
meta_dir = "data/metadata/"

#img_list = listdir(img_dir)
#N = len(img_list)
N = 535234
list_random = range(N)
random.shuffle(list_random)

# finding images that metadata exists
meta_avail = np.zeros(N, dtype=bool)
for i in range(N):
    meta_fname = os.path.join(meta_dir,('%05d.json'%(i+1)))
    if os.path.isfile(meta_fname):
        meta_avail[i] = True

# assign validataion set
valset = np.zeros(N, dtype=bool)
n_valset = int(round(N*0.1))
count = 0
random.shuffle(list_random)
for i in range(N):
    idx = list_random[i]
    if meta_avail[idx]:
        valset[idx]=True
        count = count + 1
        if count == n_valset:
            break

# writing out to textfile
train_f = open('random_train.txt','w')
val_f = open('random_val.txt','w')
for i in range(N):
    if meta_avail[i]:
        if valset[i]:
            val_f.write("%d\n" % (i+1))
        else:
            train_f.write("%d\n" % (i+1))
train_f.close()
val_f.close()
