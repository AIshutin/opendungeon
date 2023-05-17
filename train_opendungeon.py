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
from utils import *
import argparse
import logging
import time


def get_mask(inputs):
    c = ((inputs != -100) & (inputs != tokenizer.pad_token_id)).sum()
    middle = c // 2 + 1
    mx = c - 1
    state = 1
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    dm_cnt = 0

    for tj in range(0, mx):
        if inputs[tj] == DM_id:
            dm_cnt += 1
            state = 0
            mask[tj] = 0
            continue
        elif inputs[tj] == P_id:
            state = 1
            if tj > 0 and mask[tj - 1] == 1:
                mask[tj] = 1
            continue

        if dm_cnt < 2:
            continue
        if state != 0:
            continue
        mask[tj] = 1
    
    return mask


class DMTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # implement custom logic here
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        
        
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for sentence in range(labels.shape[0]):
            mask[sentence] = get_mask(labels[sentence])
        
        logits = logits[mask]
        labels = labels[mask]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def visualise_mask(inputs, mask, label=None):
    words = [tokenizer.decode(inputs[i]).lower() for i in range(len(mask))] # , skip_special_tokens=True
    for i in range(len(mask)):
        if mask[i] == 0:
            words[i] = words[i].upper()
    if label is not None:
        wandb.log({label: ' '.join(words)})
    print(' '.join(words))

def calc_acc_online(top1_arr, top3_arr, top5_arr, top10_arr, true_prob, inputs):
    inputs = inputs[1:] # skip bos
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    total = 0
    loss = 0
    c = ((inputs != -100) & (inputs != tokenizer.pad_token_id)).sum()
    middle = c // 2 + 1
    mx = c - 1
    state = 1
    mask = []
    
    alternative_inputs = inputs.copy()
    
    for tj in range(0, mx):
        if inputs[tj] == DM_id:
            state = 0
        elif inputs[tj] == P_id:
            state = 1
        if tj < middle:
            mask.append(1)
            continue
        mask.append(state)
        if inputs[tj] == DM_id:
            continue
        if state != 0 and inputs[tj] != P_id:
            continue
        
        total += 1
        
        fix = inputs[tj]
        
        loss  += -np.log(true_prob[tj])
        top1  += fix in top1_arr[tj]
        top3  += fix in top3_arr[tj]
        top5  += fix in top5_arr[tj]
        top10 += fix in top10_arr[tj]
        alternative_inputs[tj] = top1_arr[tj]
             
    assert(isinstance(top1, int))
    assert(isinstance(top10, int))
    assert(isinstance(total, int))

    return {"acc-top1": top1, "acc-top3": top3, "acc-top5": top5, "acc-top10": top10, "total": total, 'loss': loss}


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    top10_idx = preds['top10_idx']
    top1_idx = preds['top1_idx']
    true_prob = preds['true_prob']
    acc10 = 0
    acc1 = 0
    acc5 = 0
    acc3 = 0
    total = 0
    loss = 0
    for i in range(top10_idx.shape[0]):
        acc = calc_acc_online(top1_idx[i],  preds['top3_idx'][i], preds['top5_idx'][i],
                              top10_idx[i], true_prob[i],         labels[i])
        acc10 += acc['acc-top10']
        acc5 += acc['acc-top5']
        acc3 += acc['acc-top3']
        acc1 += acc['acc-top1']
        loss += acc['loss']
        total += acc['total']
    return {"acc10": acc10 / total, "acc1": acc1 / total, "acc5": acc5 / total, "acc3": acc3 / total, 
            "loss_specific": loss / total}


def check_for_leaks(train_dataset, test_dataset):
    decoded_train = []
    decoded_train_string = ""
    for train in train_dataset:
        text = tokenizer.decode(train['input_ids'])
        decoded_train.append(text)
    decoded_train_string = '---'.join(decoded_train)
    
    for el in tqdm(test_dataset):
        el = el['input_ids']
        phrases = []
        mask = get_mask(el)
        curr_phrase = []
        for i in range(1, len(mask)):
            if mask[i] == 1:
                curr_phrase.append(el[i])
            elif len(curr_phrase) != 0:
                phrases.append(tokenizer.decode(curr_phrase))
        if len(curr_phrase) != 0:
            phrases.append(tokenizer.decode(curr_phrase))
        
        for phrase in set(phrases):
            if phrase not in decoded_train_string:
                continue
            for text in decoded_train:
                if phrase in text and len(phrase) > 30:
                    print('Phrase', phrase)
                    print('TEXT', text)
                    break

                    
class GenerateTextCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        model.eval()
        batch = tokenizer(tokenizer.bos_token + ' ' + prompt, return_tensors='pt')
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=50, bad_words_ids=[[DM_id]], top_k=1, min_new_tokens=10)
        output_str = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print('GENERATED: ', output_str)
        print('VISUALISED_MASK: ')
        visualise_mask(output_tokens[0], ~get_mask(output_tokens[0]))
        
        wandb.log({'eval/generated': output_str})       

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='OpenDungeon trainer')
    parser.add_argument("--model", default="EleutherAI/gpt-j-6B", help="huggingface model (space/name)")
    parser.add_argument("--int8",  action='store_true', help="use 8bit mode instead of 16")
    parser.add_argument("--disable_tqdm",  action='store_true', help="disable progressbars")
    parser.add_argument("--tags", nargs='+', help="tags for wandb")
    parser.add_argument("--dropout_p", type=float, default=0.1, help="dropout proba in frozen model")
    parser.add_argument("--lora_dropout_p", type=float, default=0.05, help="dropout proba in LoRA")
    parser.add_argument("--lora_alpha", type=float, default=32, help="LoRA alpha parameter for init distribution")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank. Set to 0 to disable LoRA.")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--setup_pad_token",  action='store_true')
    parser.add_argument("--output_dir", default='critical_role_test')
    parser.add_argument("--output_path", default='/extra_disk_1/yozh/')
    parser.add_argument("--val_subsample", default=None, type=float)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.setup_pad_token:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    else:
        logging.warning("Pad token is not setup manually. Check if it's already setup in tokenizer.")
    DM_id = tokenizer.encode("DM")[0]
    P_id = tokenizer.encode("Player")[0]
    tokenizer.eos_token_id
    
    train_dataset, val_dataset = get_datasets(tokenizer)
    
    if args.val_subsample is not None:
        if args.val_subsample < 1.0:
            raise NotImplemented()
        else:
            val_dataset = torch.utils.data.Subset(val_dataset, list(range(int(args.val_subsample))))

    if not args.int8:
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                            torch_dtype=torch.float16,
                                            device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                            load_in_8bit=True,
                                            device_map='auto')
    
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    
    activate_dropout(model, args.dropout_p)
    
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout_p,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)  
    
    model_wrapped_for_trainer = EfficientLoRASaver(model)
    state_dict_adapters_only = model_wrapped_for_trainer.state_dict()
    model_wrapped_for_trainer.load_state_dict(state_dict_adapters_only)
    
    wandb.init(project="opendungeon", 
               tags=args.tags,
               config=args)
    wandb.run.log_code('.')
    
    t0 = time.time()
    
    trainer = DMTrainer(
        model=model_wrapped_for_trainer, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps, 
            max_steps=args.max_steps, 
            learning_rate=args.lr, 
            fp16=True,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            output_dir=args.output_path + args.output_dir, 
            report_to="wandb",
            evaluation_strategy="steps",
            disable_tqdm=args.disable_tqdm
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[GenerateTextCallback]
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    
    print(f'Time elapsed since init: {time.time() - t0:.2f} seconds')
    print(f'Max GPU memory allocated {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB')
    
    model.eval()
    batch = tokenizer(prompt, return_tensors='pt')

    with torch.cuda.amp.autocast():
      output_tokens = model.generate(**batch, max_new_tokens=50)

    print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

