import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import set_seed


def nop(it, *a, **k):
    return it

tqdm.tqdm = nop
device = "cuda" if torch.cuda.is_available() else "cpu"

class Gen_Model:
    def __init__(self, model_checkpoint, device = device, seed = 21):
        set_seed(seed)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, truncation_side = "left", use_fast = False, model_max_length = 1024)
        self.pipe = pipeline(task = "text-generation",model = self.model, tokenizer = self.tokenizer)
        