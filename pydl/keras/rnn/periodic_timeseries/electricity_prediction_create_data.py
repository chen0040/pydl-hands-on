import numpy as np
import matplotlib.pyplot as plt
import os
import re

VERY_LARGE_DATA_DIR = '../../../very_large_data'
DATA_DIR = '../../../data'

# this file is not included in the git due to its size, but can be downloaded
# from https://archive.ics.uci.edu/ml/machine-learning-databases/00321/
fid = open(os.path.join(VERY_LARGE_DATA_DIR, 'LD2011_2014.txt'), 'rt')
data = []
cid = 250 # for customer with id 250
for line in fid:
    if line.startswith("\"\";"):
        continue
    cols = [float(re.sub(",", ".", x)) for x in line.strip().split(";")[1:]]
    data.append(cols[cid])
fid.close()

NUM_ENTRIES = 1000
plt.plot(range(NUM_ENTRIES), data[0:NUM_ENTRIES])
plt.ylabel("electricity consumption")
plt.xlabel("time (1pt = 15 mins)")
plt.show()

np.save(os.path.join(DATA_DIR, "LD_250.npy"), np.array(data))
