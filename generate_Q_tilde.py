import torch
from argparse import ArgumentParser
from tqdm import tqdm
import json
import jsonlines
from generator import Gen_Model
from prompt import Promptconfig
from dataclasses import dataclass
from generation_prefix import retrieve_prompt_prefix
from generator import Gen_Model

@dataclass
class Tilde_Config:
    max_length: int = 64
    do_sample : bool = True
    temperature: float = 0.5
    top_p: float = 1.0
    return_dict_in_generate : bool = True
    output_scores : bool = True
    num_return_sequences: int = 1

@dataclass
class GlobalConfig:
    expt_name : str = "Tilde production"
    seed : int = -10
    dataset_name : str = "com2sense"

class Generate_Negation:
    def __init__(self, args):
        self.args = args
        self.general_prefix = open("promptsv_0_1/strategyqa/q_to_sent.txt", "r").read()
        self.prompt_prefix_dict = retrieve_prompt_prefix(args.dataset_name)
        self.prompt_configs = Promptconfig(self.prompt_prefix_dict["abductive"], self.prompt_prefix_dict["belief"],
                                           self.prompt_prefix_dict["negation"], self.prompt_prefix_dict["question"],
                                           self.prompt_prefix_dict["nli"])
        self.gen_model = Gen_Model("google/flan-t5-xl", seed=args.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prompt_sent(self,Q):
        generation_config = Tilde_Config()
        generation_config.temperature = self.args.temp
        prompt_str = self.create_general_sentence(Q)
        # print(f"Prompt str = {prompt_str}")
        num_Es_to_generate = generation_config.num_return_sequences
        E_list = []
        while len(E_list) < num_Es_to_generate:
            generation_config.num_return_sequences = num_Es_to_generate - len(E_list)
            response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
            response = self.gen_model.model.generate(**response_inputs, **generation_config.__dict__)
            response = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
            E_list.extend(Generate_Negation.filter_generated_explanations(response))
        # print(f"E_Tilde = {E_list[0]}")
        return E_list[0]

    def prompt_tilde(self,Q):
        generation_config = Tilde_Config()
        generation_config.temperature = self.args.temp
        prompt_str = self.prompt_configs.create_new_negation_prompt(Q)
        # print(f"prompt str = {prompt_str}")
        num_Es_to_generate = generation_config.num_return_sequences
        E_list = []
        while len(E_list) < num_Es_to_generate:
            generation_config.num_return_sequences = num_Es_to_generate - len(E_list)
            response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
            response = self.gen_model.model.generate(**response_inputs, **generation_config.__dict__)
            response = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
            E_list.extend(Generate_Negation.filter_generated_explanations(response))
        # print(f"E_Tilde = {E_list[0]}")
        return E_list[0]

    @staticmethod
    def filter_generated_explanations(explanations):
        filtered_explanations = [explanation.strip() for explanation in explanations]
        filtered_explanations = list(filter(lambda exp: len(exp) > 0 and exp.endswith("."), filtered_explanations))
        filtered_explanations = [explanation[0].upper() + explanation[1:] for explanation in filtered_explanations]
        if len(filtered_explanations) == 0:
            filtered_explanations.append(explanations[0].strip())
        filtered_explanations = list(dict.fromkeys(filtered_explanations))

        return filtered_explanations
    
    def create_general_sentence(self, Q):
        return f"{self.general_prefix}" \
            f"Q: {Q}\n"\
                f"A:"

        


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--dataset_name", default = "com2sense", type = str)
    args.add_argument("--seed", default= 42, help=" Generation seed", type= int)
    args.add_argument("--temp", default = 0.5, type = float, help = "Generation temperature")

    args = args.parse_args()
    args.mode = "normal"

    if args.dataset_name == "com2sense":
        if args.mode == "normal":
            args.data_filename =  f"./data/{args.dataset_name}/dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_Q_Q_tilde_{args.seed}"
    elif args.dataset_name == "csqa2":
        if args.mode == "normal":
            args.data_filename = f"./data/{args.dataset_name}/CSQA2_dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_Q_Q_tilde_{args.seed}"
    elif args.dataset_name == "creak":
        if args.mode == "normal":
            args.data_filename = f"./data/{args.dataset_name}/dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_Q_Q_tilde_{args.seed}"
    elif args.dataset_name == "strategyqa":
        print("Strategy QA does not have tilde mode")
        if args.mode == "normal":
            args.data_filename = f"./data/{args.dataset_name}/dev.json"
            args.out_filename = f"./data/{args.dataset_name}/dev_Q_Q_tilde_{args.seed}.json"


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

    model = Generate_Negation(args)
    ExptConfig = GlobalConfig(
        seed = args.seed,
        dataset_name = args.dataset_name
    )

    print(f"The cofig for this experiment = {ExptConfig}")

    datas = []

    for sample_idx, sample in tqdm(enumerate(samples), total = len(samples)):
        if args.dataset_name == "com2sense":
            if args.mode == "normal":
                Q_tilde = model.prompt_tilde(sample["sent"]) 
        elif args.dataset_name == "csqa2":
            if args.mode == "normal":
                Q_tilde = model.prompt_tilde(sample["question"])
        elif args.dataset_name == "creak":
            if args.mode == "normal":
                Q_tilde = model.prompt_tilde(sample["sentence"])
        elif args.dataset_name == "strategyqa":
            if args.mode == "normal":
                # Q_tilde = model.prompt_tilde(sample["question"])
                Q_general = model.prompt_sent(sample["question"])
                Q_tilde = model.prompt_tilde(Q_general)
                data = {}
                data["Q"] = Q_general
                data["Q_tilde"] = Q_tilde
                data["Q_orig"] = sample["question"]
                data["A"] = sample["answer"]
                datas.append(data)
                # print(f"Q = {sample['question']}")
                # print(f"Q_tilde = {Q_tilde}")
                # print(f"Q general sentence = {Q_general}")


    with open(args.out_filename, "w") as fp:
        json.dump(datas, fp)
        

        

    