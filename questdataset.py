import torch
import requests
import random
import os
import shutil
import json
from os.path import expanduser


def download_file(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)


class QuestDataset(torch.utils.data.Dataset):
    REMOTE_URL = "https://huggingface.co/datasets/OpenDungeon/chooseyourstory/resolve/main/"
    CACHE_NAME = "opendungeon/chooseyourstory"
    RETRY_NODE_CNT = 3
    MIN_DOC_LENGTH = 600
    MAX_DOC_LENGTH = 1900
    MIN_PHRASES = 4
    BUDGET = 200
    
    def __init__(self, datadir, tokenizer, player="Player", dm="DM", prefix="It is a fantasy role-play game.\n\n", 
                       cache_dir=None, tokenize=False):
        if cache_dir is None:
            cache_dir = os.getenv("HF_DATASETS_CACHE", os.path.join(expanduser("~"), ".cache/huggingface/datasets"))

        if not os.path.exists(datadir): # use remote
            if datadir in {"data_train", "data_test", "data_val"}:
                path = os.path.join(cache_dir, self.CACHE_NAME, datadir)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    filename = os.path.join(cache_dir, self.CACHE_NAME, datadir + '.zip')
                    download_file(self.REMOTE_URL + datadir + ".zip", filename)
                    shutil.unpack_archive(filename, os.path.join(cache_dir, self.CACHE_NAME))
                datadir = path
                
            else:
                raise Exception(f"no path or remote data: {datatir}")

        files = [os.path.join(datadir, el) for el in os.listdir(datadir)]
        self.states = {}
        self.states_list = []

        for fn in os.listdir(datadir):
            quest = json.load(open(os.path.join(datadir, fn)))
            for key, state in quest.items():
                state['next_nums'] = [fn + "/" + str(el) for el in state['next_nums']]
                state['main_text'] = state['main_text']
                self.states[fn + "/" + key] = state
                self.states_list.append(fn + "/" + key)
        
        self.player = player
        self.dm = dm
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.tokenize = tokenize
    
    def __len__(self):
        return len(self.states)
    
    def check(self, text):
        phrases = text.count(self.dm) + text.count(self.player)
        length = len(self.tokenizer(self.prefix + text)['input_ids'])
        return self.MIN_PHRASES <= phrases and self.MIN_DOC_LENGTH <= length <= self.MAX_DOC_LENGTH
    
    def pick_choices(self, node, budget, text=""):
        old_text = text
        text = text + self.dm + ': ' + self.states[node]['main_text'] + '\n'
        text_len = len(self.tokenizer(self.prefix + text)['input_ids'])
        
        if text_len > self.MAX_DOC_LENGTH:
            if self.check(old_text):
                return old_text, budget
        
        for _ in range(self.RETRY_NODE_CNT):
            budget -= 1
            if budget <= 0:
                if self.check(text):
                    return text, budget
                else:
                    return None, budget
            i = random.randint(0, len(self.states[node]['next_nums']))
            if i == len(self.states[node]['next_nums']):
                new_text = text
            else:
                new_text = text + self.player + ": " + self.states[node]['query_texts'][i] + '\n'
                new_text, budget = self.pick_choices(self.states[node]['next_nums'][i], budget, new_text)
            if new_text is not None and self.check(new_text):
                return new_text, budget
        return None, budget
    
    def try_to_extract(self, i, tokenize=False):
        for _ in range(self.RETRY_NODE_CNT):
            text, _ = self.pick_choices(self.states_list[i], self.BUDGET)
            
            if text is not None:
                return self.prefix + text
    
    def __getitem__(self, i):
        fails = 0
        while True:
            text = self.try_to_extract(i, tokenize=True)
            if text is None:
                i = random.randint(0, len(self) - 1)
                fails += 1
            else:
                if self.tokenize:
                    out = self.tokenizer(text)
                    assert(len(out["input_ids"]) <= self.MAX_DOC_LENGTH)
                    return out
    
                return text


if __name__ == "__main__":
    import transformers

    random.seed(0) 
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    mydataset = QuestDataset("data_test", tokenizer)
    
    print('state', mydataset.states_list[100])
    print(mydataset[100])
