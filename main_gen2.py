import os
import pickle
import jsonlines
import json
from argparse import ArgumentParser
from tqdm import tqdm
from dataclasses import dataclass

from generate_tree import GenerationWrapper
from generation_prefix import Generation_Prefix

@dataclass
class Global_Config:
    dataset_name :str = "com2sense"
    mode :str = "normal"
    tree_lib : str = "treelib"
    seed : int = 42



if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--dataset_name", default = "com2sense", type = str)
    args.add_argument("--mode", default="normal", help = "define the mode")
    args.add_argument("--tree", default="treelib", help = "The library to use")
    args.add_argument("--seed", default= 42, help=" Generation seed", type= int)
    # args.add_argument("--start", default = 0, help = "Start index of the dataset", type=int)
    # args.add_argument("--end", default= -10, help="End index of the dataset", type = int)
    args = args.parse_args()
    if args.dataset_name == "com2sense":
        if args.mode == "normal":
            # args.data_filename =  f"./data/{args.dataset_name}/dev.json"
            args.data_filename =  f"./data/{args.dataset_name}/dev_modified.json" ## we put the modified file
            # args.out_filename = f"./data/{args.dataset_name}/dev_G_normal.pkl_{args.seed}"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_normal_modified.pkl_{args.seed}"
        elif args.mode == "tilde":
            args.data_filename = f"./data/{args.dataset_name}/bothdev.Q.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_tilde.pkl_{args.seed}"
    elif args.dataset_name == "csqa2":
        if args.mode == "normal":
            args.data_filename = f"./data/{args.dataset_name}/CSQA2_dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_normal.pkl_{args.seed}"
        else:
            args.data_filename = f"./data/{args.dataset_name}/dev.Q.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_tilde.pkl_{args.seed}"
    elif args.dataset_name == "creak":
        if args.mode == "normal":
            args.data_filename = f"./data/{args.dataset_name}/dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_normal.pkl_{args.seed}"
        elif args.mode == "tilde":
            args.data_filename = f"./data/{args.dataset_name}/dev.Q.json"
            args.out_filename = f"data/{args.dataset_name}/dev_G_tilde.pkl_{args.seed}"
    elif args.dataset_name == "strategyqa":
        if args.mode == "normal":
            args.data_filename = f"./data/{args.dataset_name}/dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_normal.pkl_{args.seed}"
        elif args.mode == "tilde":
            args.data_filename = f"./data/{args.dataset_name}/dev_Q_Q_tilde_{args.seed}.json"
            args.out_filename = f"data/{args.dataset_name}/dev_G_tilde.pkl_{args.seed}"

    prompt_prefix = Generation_Prefix()
    prompt_prefix_dict = prompt_prefix.retrieve_prompt_prefix(args.dataset_name)
    generator = GenerationWrapper(prompt_prefix_dict["abductive"], prompt_prefix_dict["belief"],
                                  prompt_prefix_dict["negation"], prompt_prefix_dict["question"], prompt_prefix_dict["nli"],
                                   seed = args.seed,depth=2)
    
    if args.mode == "normal":
        if args.dataset_name == "csqa2":
            with jsonlines.open(args.data_filename, "r") as fp:
                samples = list(fp)
        elif args.dataset_name == "com2sense":
            with open(args.data_filename) as fp:
                samples = json.load(fp)
        elif args.dataset_name == "creak":
            with jsonlines.open(args.data_filename, "r") as fp:
                samples = list(fp)
        elif args.dataset_name == "strategyqa":
            with open(args.data_filename) as fp:
                samples = json.load(fp)
    elif args.mode == "tilde":
        if args.dataset_name == "strategyqa":
            with open(args.data_filename) as fp:
                samples = json.load(fp)
        else:        
            with jsonlines.open(args.data_filename, "r") as fp:
                samples = list(fp)

    if os.path.exists(args.out_filename):
        with open(args.out_filename, "rb") as f:
            orig_out_list = pickle.load(f)
    else:
        orig_out_list = []

    out_list = orig_out_list

    #################### Config details ###################
    config = Global_Config(
        dataset_name=args.dataset_name,
        mode= args.mode,
        tree_lib= args.tree,
        seed = args.seed
    )
    print(f"The generation config for this experiment : {config}")


    #################### Generation Procedure #####################

    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples)):
        if args.dataset_name == "com2sense":
            if args.mode == "normal":
                # G = generator.create_graph(sample["sent"])
                G = generator.create_graph(sample["Q"]) ## replaced by modified
            else:
                G = generator.create_graph(sample["Q"], sample["Q_tilde"]) 
        elif args.dataset_name == "csqa2":
            if args.mode == "normal":
                G = generator.create_graph(sample["question"])
            else:
                G = generator.create_graph(sample["Q"], sample["Q_tilde"]) 
        elif args.dataset_name == "creak":
            if args.mode == "normal":
                G = generator.create_graph(sample["sentence"])
            else:
                G = generator.create_graph(sample["Q"], sample["Q_tilde"])
        elif args.dataset_name == "strategyqa":
            if args.mode == "normal":
                G = generator.create_graph(sample["question"])
            elif args.mode == "tilde":
                print(f"Q = {sample['Q']}")
                print(f"Q_tilde = {sample['Q_tilde']}")
                G = generator.create_graph(sample["Q"], sample["Q_tilde"])
        out_list.append(G)
        if sample_idx % 100 == 0: 
            with open(args.out_filename, "wb") as f:
                pickle.dump(out_list, f)

    with open(args.out_filename, "wb") as f:
        pickle.dump(out_list, f)