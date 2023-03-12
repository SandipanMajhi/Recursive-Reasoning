import pickle
from argparse import ArgumentParser, Namespace

import jsonlines
import json
import torch
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, AutoTokenizer

from infer import Inference_Wrapper
from roberta_verifier import NLIModel

def parse_args():
    args = ArgumentParser()
    args.add_argument("--dataset_name", default= "com2sense", type=str, help= "provide the name of the dataset")
    args = args.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # args.data_filename = f"./data/{args.dataset_name}/bothdev.Q.json" 
    args.data_filename = f"./data/{args.dataset_name}/dev.json"
    args.out_file = f"./data/{args.dataset_name}/dev.G.pkl"
    return args

if __name__ == "__main__":
    args = parse_args()
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast = False)
    Inference_Wrapper.nli_model = NLIModel(model, tokenizer)

    # with jsonlines.open(args.data_filename, "r") as f:
    #     samples = list(f)

    with open(args.data_filename) as fp:
        samples = json.load(fp)

    with open(args.out_file, "rb") as fp:
        G_samples = pickle.load(fp)

    acc_result = [0,0]
    for sample, G in tqdm(zip(samples, G_samples), total= len(samples)):
        if G.size() == 1:
            inferred_answer = 1 if G["Q"].data["blf"][0] >= G["Q"].data["blf"][1] else -1
        elif G.size() > 1:
            score_list, correct_E_dict, graphsat, belief, consistency = Inference_Wrapper.infer(G)
            sum_score = sum([score[1] for score in score_list])
            inferred_answer = 1 if sum_score >= 0 else -1
        else:
            inferred_answer = 1

        gt_answer = 1 if sample["label"] == "True" else -1
        acc_result[0 if inferred_answer == gt_answer else 1] += 1

    # for sample, G in tqdm(zip(samples, G_samples), total= len(samples)):
    #     if G.size() == 1:
    #         inferred_answer = 1 if G["Q"].data["blf"][0] >= G["Q"].data["blf"][1] else -1
    #     elif G.size() > 1:
    #         score_list, correct_E_dict, graphsat, belief, consistency = Inference_Wrapper.infer(G)
    #         sum_score = sum([score[1] for score in score_list])
    #         inferred_answer = 1 if sum_score >= 0 else -1
    #     else:
    #         inferred_answer = 1

    #     gt_answer = 1 if sample["A"] else -1
    #     acc_result[0 if inferred_answer == gt_answer else 1] += 1
    print(f"Acc : {acc_result}")
