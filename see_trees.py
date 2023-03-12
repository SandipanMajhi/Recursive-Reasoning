import pickle as pkl
from tqdm import tqdm

datapath = "data/strategyqa/dev_G_normal.pkl_21"

with open(datapath, "rb") as fp:
    samples = pkl.load(fp)

i = 0

for G in tqdm(samples):
    print(G)
    i += 1
    if i == 30:
        break
