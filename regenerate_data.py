from generator import Gen_Model
from argparse import ArgumentParser
import json
import jsonlines
import torch
from dataclasses import dataclass
from tqdm import tqdm
import pickle as pkl
import re

# @dataclass
# class GenerationConfig:
#     max_length: int = 64
#     do_sample : bool = True
#     temperature: float = 1.0
#     top_p: float = 1.0
#     return_dict_in_generate : bool = True
#     output_scores : bool = True
#     num_return_sequences: int = 1

@dataclass
class GenerationConfig:
    max_length: int = 128
    do_sample : bool = False
    num_beams : int = 2
    return_dict_in_generate : bool = True
    output_scores : bool = True
    num_return_sequences: int = 1


def create_regen_prompt(prefix, sentence):
    return  f"{prefix}\n\n" \
            f"Q: {sentence}\n"\
            f"A:"

def model_generate(gen_model,sentence, prefix, device):
    generation_config = GenerationConfig()
    prompt_str = create_regen_prompt(prefix, sentence)
    response_inputs = gen_model.tokenizer(prompt_str, return_tensors = "pt").to(device)
    response = gen_model.model.generate(**response_inputs, **generation_config.__dict__)
    response = gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
    if "The statement is true." in response[0]:
        response = re.sub("The statement is true.","",response)
        response = response.strip()
        response = response[0].upper() + response[1:]

    if "The sentence is true." in response[0]:
        response = re.sub("The sentence is true.","",response)
        response = response.strip()
        response = response[0].upper() + response[1:]
    # print(response[0])
    return response[0]


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--dataset_name", default = "com2sense", type=str, help= "provide the name of the dataset")
    args.add_argument("--split_name", default = "dev", type = str, help = "The Data Split of the dataset")
    args.add_argument("--prefix_text", default= "promptsv_0_1/com2sense/q_to_sent.txt", type = str, help = "The prefix of the regeneration")
    args.add_argument("--seed", default=21, type=int, help="Generation seed")

    args = args.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_model = Gen_Model("google/flan-t5-xl", device=args.device, seed=args.seed)

    if args.dataset_name == "csqa2":
        args.data_filename = f"./data/{args.dataset_name}/CSQA2_dev.json"
        with jsonlines.open(args.data_filename, "r") as fp:
            samples = list(fp)
    elif args.dataset_name == "com2sense":
        args.data_filename =  f"./data/{args.dataset_name}/dev.json"
        with open(args.data_filename) as fp:
            samples = json.load(fp)
    elif args.dataset_name == "creak":
        args.data_filename = f"./data/{args.dataset_name}/dev.json"
        with jsonlines.open(args.data_filename, "r") as fp:
            samples = list(fp)
    elif args.dataset_name == "strategyqa":
        args.data_filename = f"./data/{args.dataset_name}/dev.json"
        with open(args.data_filename) as fp:
            samples = json.load(fp)

    prompt_prefix = open(args.prefix_text, "r").read()

    out_list = []
    args.out_filename = f"./data/{args.dataset_name}/dev_modified.json"
    
    ######################## Generation Procedure ###############################

    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples)):
        if args.dataset_name == "csqa2":
            out_list.append({
                "Q" : model_generate(gen_model= gen_model, sentence= sample["question"], prefix = prompt_prefix, device=args.device),
                "label" : sample["answer"]
            })
        elif args.dataset_name == "com2sense":
            print(sample["sent"])
            out_list.append({
                "Q" : model_generate(gen_model= gen_model, sentence= sample["sent"], prefix = prompt_prefix, device=args.device),
                "label" : sample["label"]
            })
        elif args.dataset_name == "creak":
            out_list.append({
                "Q" : model_generate(gen_model= gen_model, sentence= sample["sentence"], prefix = prompt_prefix, device=args.device),
                "label" : sample["label"]
            })
        elif args.dataset_name == "strategyqa":
            out_list.append({
                "Q" : model_generate(gen_model= gen_model, sentence= sample["question"], prefix = prompt_prefix, device=args.device),
                "label" : sample["answer"]
            })

            
    with open(args.out_filename, "w") as f:
        json.dump(out_list, f)
    


