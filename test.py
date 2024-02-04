import pandas as pd
import numpy as np
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.nn.parameter import Parameter
from sklearn import preprocessing
from sklearn import model_selection
import time
from typing import List, Dict, Tuple, Optional
import warnings
from tqdm import tqdm
from sklearn import preprocessing
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
label_mapping = {'O': 0, 'B-ORG': 1, 'B-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-ORG': 6, 'I-MISC': 7, 'I-LOC': 8}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
with open('test_data.csv') as f:
	data = f.read().splitlines()
df = pd.DataFrame({"text": data})
def find_blank_row_indices(df, column_name='text'):
    blank_indices = df.index[df[column_name].isnull() | (df[column_name] == ' ')].tolist()
    return blank_indices

def rows_to_sentences_and_labels(df):
    sentences = []
    blank = []
    start_end = []
    current_sentence = []
    current_blank = []
    current_start_end = []
    for index, row in tqdm(df.iterrows(), total = len(df)):
        word = row['text']
        if index==0 or df.iloc[index-1]['text'].strip()=='.':
            current_start_end.append(index)
            if word.strip() == '':
                current_blank.append(index)
            else:
                current_sentence.append(word.strip())
        elif word.strip() =='.':
            current_sentence.append(word.strip())
            current_start_end.append(index)
            sentences.append(current_sentence)
            start_end.append(current_start_end)
            blank.append(current_blank)
            current_start_end = []
            current_sentence = []
            current_blank = []
        elif word.strip()=="":
            current_blank.append(index)
        elif word.strip()!="":
            current_sentence.append(word.strip())
    return sentences, blank, start_end
sentences, blank, start_end = rows_to_sentences_and_labels(df)   
class CFG:
    seed = 42
    max_len = 1024
    train_bs = 8
    valid_bs = 8
    epochs = 5
    model_name = "microsoft/deberta-v3-small"
def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

set_seed(CFG.seed)


class EntityDataset:
    def __init__(self, tokenizer, texts):
        self.texts = texts
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        ids = []
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(
                s,
                add_special_tokens=False,
                max_length = 1024
            )
            input_len = len(inputs)
            if input_len>1:
                ids.append(inputs[0])
            elif input_len==1:
                ids.extend(inputs)
            else:
                continue
        ids = ids[:CFG.max_len - 2]
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        mask = [1] * len(ids)
        padding_len = CFG.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.config = AutoConfig.from_pretrained("microsoft/deberta-v3-large", output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        self.model = AutoModel.from_pretrained("microsoft/deberta-v3-large", config=self.config)
        self.model.gradient_checkpointing_enable()
        self.fc = nn.Linear(self.config.hidden_size*4, self.num_tag)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, 
        ids, 
        mask, 
        target_tag=None
    ):
        output = self.model(
            input_ids=ids,attention_mask=mask,
            output_hidden_states=True
        )
        all_hidden_states = torch.stack(output.hidden_states)
        last_four_hidden_states = all_hidden_states[-4:]
        concatenated_hidden_states = torch.cat(tuple(last_four_hidden_states), dim=-1)

        tag = self.fc(concatenated_hidden_states)
        if target_tag is not None:
            loss = loss_fn(tag, target_tag, mask, self.num_tag)
            return tag, loss
        return tag

class EntityModel1(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel1, self).__init__()
        self.num_tag = num_tag
        self.config = AutoConfig.from_pretrained("microsoft/deberta-v3-base", output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base", config=self.config)
        self.model.gradient_checkpointing_enable()
        self.fc = nn.Linear(self.config.hidden_size*4, self.num_tag)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, 
        ids, 
        mask, 
        target_tag=None
    ):
        output = self.model(
            input_ids=ids,attention_mask=mask,
            output_hidden_states=True
        )
        all_hidden_states = torch.stack(output.hidden_states)
        last_four_hidden_states = all_hidden_states[-4:]
        concatenated_hidden_states = torch.cat(tuple(last_four_hidden_states), dim=-1)

        tag = self.fc(concatenated_hidden_states)
        if target_tag is not None:
            loss = loss_fn(tag, target_tag, mask, self.num_tag)
            return tag, loss
        return tag

class EntityModel2(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel2, self).__init__()
        self.num_tag = num_tag
        self.config = AutoConfig.from_pretrained("microsoft/deberta-v3-small", output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        self.model = AutoModel.from_pretrained("microsoft/deberta-v3-small", config=self.config)
        self.model.gradient_checkpointing_enable()
        self.fc = nn.Linear(self.config.hidden_size*4, self.num_tag)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, 
        ids, 
        mask, 
        target_tag=None
    ):
        output = self.model(
            input_ids=ids,attention_mask=mask,
            output_hidden_states=True
        )
        all_hidden_states = torch.stack(output.hidden_states)
        last_four_hidden_states = all_hidden_states[-4:]
        concatenated_hidden_states = torch.cat(tuple(last_four_hidden_states), dim=-1)

        tag = self.fc(concatenated_hidden_states)
        if target_tag is not None:
            loss = loss_fn(tag, target_tag, mask, self.num_tag)
            return tag, loss
        return tag


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs)) 


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

def valid_fn(val_loader, model, model1, model2, device):
    model.eval()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Val')
    valid_preds = []
    for _, data in pbar:
        ids   =  data['input_ids'].to(device, dtype = torch.long)
        mask  =  data['attention_mask'].to(device, dtype = torch.long)
        batch_size = ids.size(0)
        with torch.no_grad():
            tag = model(ids, mask)
            tag1 = model1(ids, mask)
            tag2 = model2(ids, mask)
            final_tag = (tag+tag1+tag2)/3
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        valid_preds.append(final_tag.softmax(-1).to('cpu').numpy())
        torch.cuda.empty_cache()
        gc.collect()
    valid_preds = np.concatenate(valid_preds)
    return valid_preds
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
valid_dataset = EntityDataset(tokenizer, sentences)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=CFG.valid_bs, num_workers=5, shuffle=False, pin_memory=True
)
device = torch.device("cuda")
model = EntityModel(num_tag = 9)
checkpoint = torch.load('deberta-v3-large_epoch_5.pth')
model.load_state_dict(checkpoint)
model.to(device)

model1 = EntityModel1(num_tag = 9)
checkpoint = torch.load('deberta-v3-base_epoch_5.pth')
model1.load_state_dict(checkpoint)
model1.to(device)

model2 = EntityModel2(num_tag = 9)
checkpoint = torch.load(deberta-v3-small_epoch_4.pth')
model2.load_state_dict(checkpoint)
model2.to(device)
valid_preds = valid_fn(
    valid_data_loader,
    model,
    model1,
    model2,
    device
)
valid_preds = valid_preds.argmax(-1)
valid_preds = [[reverse_label_mapping[number] for number in sublist[1:]] for sublist in valid_preds]
final_preds = []
for valid_pred, s, b, se in zip(valid_preds, sentences, blank, start_end):
    length_sent = se[1]-se[0]+1
    valid_pred = valid_pred[:len(s)]
    if len(b)>0:
        for i in range(len(b)):
            valid_pred.insert(b[i]-se[0],'O')
    final_preds.extend(valid_pred)
blank_row_indices = find_blank_row_indices(df)
df['text'] = final_preds
df.loc[blank_row_indices, 'text'] = np.nan
df.to_csv('final_output.csv', index=False, header=False)
