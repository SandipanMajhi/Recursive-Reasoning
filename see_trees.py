import pickle as pkl
from tqdm import tqdm
from treelib import Tree

# G = Tree()
# G.create_node(tag = "Blah", identifier= "blah", parent=None, data = {
#     "name" : "Sandipan"
# })
# leaves = list(filter(lambda node : node, G.leaves()))
# print(leaves[0].data["name"])

# datapath = "data/com2sense/dev.G.pkl"
datapath = "data/com2sense/dev_G_normal_modified.pkl_21"

with open(datapath, "rb") as fp:
    samples = pkl.load(fp)

i = 0

for G in tqdm(samples):
    print(G)
    i += 1
    if i == 30:
        break
