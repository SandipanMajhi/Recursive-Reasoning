import torch
from itertools import product
from dataclasses import dataclass
from prompt import Promptconfig


@dataclass
class Generation_Config:
    max_length: int = 64
    do_sample : bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    return_dict_in_generate : bool = True
    output_scores : bool = True
    num_return_sequences: int = 3


class FlanNLI_Model:
    def __init__(self, model, tokenizer):
        self.nli_prefix = open("./promptsv_0_1/nli_prefix.txt","r").read()
        self.nli_config = Generation_Config(
            num_return_sequences=1
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

    def predict(self, premise, hypothesis):
        preds = []

        for prem, hyp in zip(premise, hypothesis):
            prompt_str = self.create_nli_prompt(prem, hyp)
            response_inputs = self.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
            response = self.model.generate(**response_inputs, **self.nli_config.__dict__)
            pred = self.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)[0]

            # print(f"Pred = {pred}")

            if pred == "entailment":
                preds.append(2)
            elif pred == "contradiction" or pred == "contradictory" or pred == "contrary":
                preds.append(0)
            else:
                preds.append(1)

        return preds

    
    def predict_from_two_worlds(self, world1, world2):
        pairs = list(product(world1, world2))
        sent1, sent2 = list(zip(*pairs))

        out1 = self.predict(sent1, sent2)
        out2 = self.predict(sent2, sent1)

        return out1, out2
    
    def create_nli_prompt(self, premise, hypothesis):
        return f"{self.nli_prefix}" \
            f"Premise: {premise}\n" \
                f"Hypothesis: {hypothesis}" \
                    f"Options: entailment, contradiction, neutral.\n" \
                        f"A:"



        