import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import wandb
import transformers
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from datasets import Dataset
import transformers
import random
import pickle
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0")
prompt = """It's a fantasy role-playing game.

DM: You are Beauregard Lionett called Beau, a human monk . You are the member of the famous group of adventurers. Previously, Ireena and a passing Crownsguard both seem to think that keeping a lot of money on their persons was pretty stupid, and under Jester's Zone of Truth Ireena denies knowing anything about it. They decide to visit Rissa's father, and to investigate Ashton and Fitz, the two gnome boys who were taunting Rissa the previous day.
DM: "No, it's just me and Syd." At this point Syd brings out the last of the food, the male gnome who you saw when you first entered. He's like, "So, all of your food is ready. Enjoy."
Player: You mentioned being able to get us friends and other type of services. Where do you find those resources?
DM: "Through a friend of mine."
Player: What's her name?
DM: "Why are you asking me all these questions? I didn't send anyone up to your room, and I'm not going to spill all of my business propositions your direction."
Player: I don't think you did any of that at all. I just think you might know who did. Even if you don't consciously know, I think you have an idea.
DM: """

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def activate_dropout(model, p):
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = p
    
def get_datasets(tokenizer):
    def tokenization(example):
        return tokenizer(tokenizer.bos_token + ' ' + example["text"])
    
    val_dataset = load_dataset("json", data_files="cr3-dm-player-solo-v1-test.json")['train']
    val_dataset = val_dataset.map(tokenization)
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_dataset = load_dataset("json", data_files="cr3-dm-player-solo-v1-train.json")['train']
    train_dataset = train_dataset.map(tokenization)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return train_dataset, val_dataset

def preprocess_logits_for_metrics(logits, inputs):
    logits = logits[..., :-1, :].contiguous()
    inputs = inputs[..., 1:].contiguous()
    
    top10_indices = logits.topk(10, dim=-1, sorted=True).indices
    top5_indices  = logits.topk(5, dim=-1, sorted=True).indices
    top3_indices  = logits.topk(3, dim=-1, sorted=True).indices
    top1_indices  = logits.topk(1, dim=-1, sorted=True).indices
    batch_size = logits.shape[0]
    vocab_size = logits.shape[2]
    inputs = inputs.unsqueeze(-1)
    probs = torch.exp(logits)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    true_prob     = torch.gather(probs, -1, inputs).squeeze(dim=-1)
    return {'top10_idx': top10_indices, 'top1_idx': top1_indices, 'top5_idx': top5_indices, 'top3_idx': top3_indices, 'true_prob': true_prob}


class EfficientLoRASaver(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, past_key_values=None, 
                attention_mask=None, token_type_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, *args, **kwargs):
        return self.model(input_ids=input_ids, past_key_values=past_key_values, 
                          attention_mask=attention_mask, token_type_ids=token_type_ids, head_mask=head_mask,
                          inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, 
                          output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                          return_dict=return_dict, *args, **kwargs)

    def state_dict(self):
        adapters = {}
        for name, module in self.model.named_modules():
          short_name = name.split('.')[-1]
          if short_name in {'lora_dropout', 'lora_A', 'lora_B'}:
            for key, w in module.state_dict().items():
              adapters[name + '.' + key] = w
        return adapters

    def load_state_dict(self, state_dict_adapters_only):
        fixed_layers = {}
        for el in state_dict_adapters_only:
          assert(el in self.model.state_dict())
        self.model.load_state_dict(state_dict_adapters_only, strict=False)
        return self
