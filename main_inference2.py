import pickle
from argparse import ArgumentParser, Namespace

import jsonlines
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass

from infer import Inference_Wrapper
from roberta_verifier import NLIModel

@dataclass
class InferConfig:
    model_name : str = "Roberta Verifier"
    dataset_name : str = "com2sense"
    mode : str = "normal"
    seed : int = 42

def parse_args():
    args = ArgumentParser()
    args.add_argument("--dataset_name", default= "com2sense", type=str, help= "provide the name of the dataset")
    args.add_argument("--mode", default="normal", type = str, help = "get the mode of the dataset")
    args.add_argument("--seed", default = 42, type=int, help="Enter the seed for the generation")
    args = args.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == "com2sense":
        if args.mode == "normal":
            args.data_filename =  f"./data/{args.dataset_name}/dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_normal.pkl_{args.seed}"
        else:
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
            args.out_filename = f"./data/{args.dataset_name}/dev_G_normal.json_{args.seed}"
        else:
            args.data_filename = f"./data/{args.dataset_name}/dev.Q.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_tilde.pkl_{args.seed}"
    elif args.dataset_name == "strategyqa":
        if args.mode == "normal":
            args.data_filename = f"./data/{args.dataset_name}/dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_G_normal.pkl_{args.seed}"
        elif args.mode == "tilde":
            args.data_filename = f"./data/{args.dataset_name}/dev_Q_Q_tilde_{args.seed}.json"
            args.out_filename = f"data/{args.dataset_name}/dev_G_tilde.pkl_{args.seed}"
    return args

if __name__ == "__main__":
    args = parse_args()
    inference_config = InferConfig(
        dataset_name=args.dataset_name,
        mode=args.mode,
        seed=args.seed
    )
    print(f"Inference Configuration : {inference_config}")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast = False)
    Inference_Wrapper.nli_model = NLIModel(model, tokenizer)

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

    with open(args.out_filename, "rb") as fp:
        G_samples = pickle.load(fp)

    print(f"data_file = {args.dataset_name}")

    acc_result = [0,0]

    for sample, G in tqdm(zip(samples, G_samples), total=len(samples)):
        if G.size() == 1:
            inferred_answer = 1 if G["Q"].data["blf"][0] >= G["Q"].data["blf"][1] else -1
        elif G.size() > 1:
            score_list, correct_E_dict, graphsat, belief, consistency = Inference_Wrapper.infer(G)
            sum_score = sum([score[1] for score in score_list])
            inferred_answer = 1 if sum_score >= 0 else -1
        else:
            inferred_answer = 1

        if args.dataset_name == "com2sense":
            if args.mode == "normal":
                gt_answer = 1 if sample["label"] == "True" else -1
            elif args.mode == "tilde":
                gt_answer = 1 if sample["A"] else -1
            acc_result[0 if inferred_answer == gt_answer else 1] += 1
        elif args.dataset_name == "csqa2":
            if args.mode == "normal":
                gt_answer = 1 if sample["answer"] == "yes" else -1
            elif args.mode == "tilde":
                gt_answer = 1 if sample["A"] else -1
            acc_result[0 if inferred_answer == gt_answer else 1] += 1
        elif args.dataset_name == "creak":
            if args.mode == "normal":
                gt_answer = 1 if sample["label"] == "true" else -1
            elif args.mode == "tilde":
                gt_answer = 1 if sample["A"] else -1
            acc_result[0 if inferred_answer == gt_answer else 1] += 1
        elif args.dataset_name == "strategyqa":
            if args.mode == "normal":
                gt_answer = 1 if sample["answer"] else -1
            elif args.mode == "tilde":
                gt_answer = 1 if sample["A"] else -1
            acc_result[0 if inferred_answer == gt_answer else 1] += 1

    print(f"Acc : {acc_result}")
