import torch
from typing import Union, List
from itertools import product

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaForSequenceClassification

class NLIModel:
    def __init__(self, model, tokenizer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

    def predict(self, premise, hypothesis):
        input_encoding = self.tokenizer(premise, hypothesis, return_tensors = "pt", padding = True)
        input_encoding = {k : v.to(self.device) for k,v in input_encoding.items()}
        with torch.no_grad():
            output = self.model(
                input_ids = input_encoding["input_ids"],
                attention_mask = input_encoding["attention_mask"]
            )

        return output.logits

    def predict_from_two_worlds(self, world1, world2):
        pairs = list(product(world1, world2))
        sent1, sent2 = list(zip(*pairs))

        out1 = self.predict(sent1, sent2)
        out1 = out1.view(len(world1), len(world2), -1).argmax(dim = -1)

        out2 = self.predict(sent2, sent1)
        out2 = out2.view(len(world1), len(world2), -1).argmax(dim=-1)

        return out1, out2

def is_in_set_T(node_name: str):
    return node_name.count("F") % 2 == 0



        