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
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import f1_score
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.simplefilter('ignore')

class CFG:
    seed = 42
    max_len = 1024
    train_bs = 4
    valid_bs = 16
    epochs = 5
    lr = 2e-5
    model_name = "microsoft/deberta-v3-base"
    only_model_name = "deberta-v3-base"

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



def init_logger(log_file='train0.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()
now = datetime.now()
datetime_now = now.strftime("%m/%d/%Y, %H:%M:%S")
LOGGER.info(f"Date :{datetime_now}")
LOGGER.info(f"trainbs: {CFG.train_bs}")
LOGGER.info(f"validbs: {CFG.valid_bs}")
LOGGER.info(f"epochs: {CFG.epochs}")
LOGGER.info(f"model_name: {CFG.model_name}")
LOGGER.info(f"max_len: {CFG.max_len}")
LOGGER.info(f"lr: {CFG.lr}")
def load_data(gt_path, data_path):
	with open(data_path) as f:
		data = f.read().splitlines()

	with open(gt_path, "r") as f:
		labels = f.read().splitlines()

	df = pd.DataFrame({"text": data, "label": labels})
	df = df[df["text"] != ";;;"]
	df["text"] = df["text"].apply(lambda x: x.replace(";;;", ""))
	df = df[~(df["label"].str.strip()=="")]
	df = df[~df["label"].str.contains(";")]

	df["label"] = df["label"].str.strip()

	df["label"] = np.where(df["label"] == "O O", "O", df["label"])
	return df

train = load_data("train_gt.csv", "train_data.csv")
valid = load_data("valid_gt.csv", "valid_data.csv")
label_mapping = {'O': 0, 'B-ORG': 1, 'B-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-ORG': 6, 'I-MISC': 7, 'I-LOC': 8}
train['label'] = train['label'].map(label_mapping)
valid['label'] = valid['label'].map(label_mapping)
ner_pos = preprocessing.LabelEncoder()
def rows_to_sentences_and_labels(df):
    sentences = []
    sentences_labels = []
    current_sentence = []
    current_labels = []

    for index, row in tqdm(df.iterrows(), total = len(df)):
        word, label = row['text'], row['label']
        current_sentence.append(word.strip())
        current_labels.append(label)
        if word.strip() == '.':
            sentences.append(current_sentence)
            sentences_labels.append(current_labels)
            current_sentence = []
            current_labels = []

    return sentences, sentences_labels

train_sentences, train_sentences_labels = rows_to_sentences_and_labels(train)
valid_sentences, valid_sentences_labels = rows_to_sentences_and_labels(valid)

class EntityDataset:
    def __init__(self, tokenizer, texts, tags):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tag = self.tags[item]
        ids = []
        target_tag =[]
        length_tag = len(tag)
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(
                s,
                add_special_tokens=False,
                max_length = 1024,
                truncation=True
            )
            input_len = len(inputs)
            if input_len>1:
                ids.append(inputs[0])
                target_tag.extend([tag[i]] * 1)
            elif input_len==1:
                ids.extend(inputs)
                target_tag.extend([tag[i]] * 1)
            else:
                continue
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        target_tag = [0] + target_tag + [0]
        mask = [1] * len(ids)

        padding_len = CFG.max_len - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        ids = ids[:CFG.max_len]
        mask = mask[:CFG.max_len]
        target_tag = target_tag[:CFG.max_len]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "length_tag": length_tag
        }




class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.config = AutoConfig.from_pretrained(CFG.model_name, output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        self.model = AutoModel.from_pretrained(CFG.model_name, config=self.config)
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

def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    torch.cuda.empty_cache()
    gc.collect()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    scaler = GradScaler()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train')
    for _, data in pbar:
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        ids   =  data['input_ids'].to(device, dtype = torch.long)
        mask  =  data['attention_mask'].to(device, dtype = torch.long)
        target_tag = data['target_tag'].to(device, dtype = torch.long)
        batch_size = ids.size(0)
        with autocast(enabled=True):
            _, loss = model(ids, mask, target_tag)
        
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_time.update(time.time() - end)
        torch.cuda.empty_cache()
        gc.collect()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{losses.avg:0.4f}',
                        lr=f'{current_lr:0.8f}',
                        gpu_mem=f'{mem:0.2f} GB')
    return losses.avg


def valid_fn(val_loader, model, device):
    losses = AverageMeter()
    model.eval()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Val')
    valid_preds = []
    valid_labels = []
    valid_length_tag = []
    for _, data in pbar:
        ids   =  data['input_ids'].to(device, dtype = torch.long)
        mask  =  data['attention_mask'].to(device, dtype = torch.long)
        target_tag = data['target_tag'].to(device, dtype = torch.long)
        length_tag = data['length_tag']
        batch_size = ids.size(0)
        with torch.no_grad():
            tag, loss = model(ids, mask, target_tag)
        losses.update(loss.item(), batch_size)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        valid_labels.append(target_tag.to('cpu').numpy())
        valid_preds.append(tag.softmax(-1).to('cpu').numpy())
        valid_length_tag.append(length_tag)
        pbar.set_postfix(eval_loss=f'{losses.avg:0.4f}',
                gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
    valid_preds = np.concatenate(valid_preds)
    valid_labels = np.concatenate(valid_labels)
    valid_length_tag = np.concatenate(valid_length_tag)
    return losses.avg, valid_preds, valid_labels, valid_length_tag



tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
train_dataset = EntityDataset(tokenizer, train_sentences, train_sentences_labels)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=CFG.train_bs, num_workers=10, shuffle=True, pin_memory=True
)
valid_dataset = EntityDataset(tokenizer, valid_sentences, valid_sentences_labels)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=CFG.valid_bs, num_workers=10, shuffle=False, pin_memory=True
)
device = torch.device("cuda")
model = EntityModel(num_tag = 9)
model.to(device)
num_train_steps = int(
    len(train_sentences) /CFG.train_bs*CFG.epochs
)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_train_steps
)
set_seed(CFG.seed)
best_f1 = 0
for epoch in range(1, CFG.epochs+1):
    LOGGER.info(f'Epoch {epoch}/{CFG.epochs}')
    
    train_loss = train_fn(train_data_loader, model, optimizer, epoch, scheduler, device)
    valid_loss, valid_preds, valid_labels, valid_length_tag = valid_fn(
        valid_data_loader,
        model,
        device
    )
    valid_preds = valid_preds.argmax(-1)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    valid_preds = [[reverse_label_mapping[number] for number in sublist[1:]] for sublist in valid_preds]
    valid_labels = [[reverse_label_mapping[number] for number in sublist[1:]] for sublist in valid_labels]

    valid_preds_trimmed = [sublist[:length] for sublist, length in zip(valid_preds, valid_length_tag)]
    valid_labels_trimmed = [sublist[:length] for sublist, length in zip(valid_labels, valid_length_tag)]
    LOGGER.info(f"Train Loss = {train_loss:.4f} Valid Loss = {valid_loss:.4f}")
    flat_valid_preds = [label for sublist in valid_preds_trimmed for label in sublist]
    flat_valid_labels = [label for sublist in valid_labels_trimmed for label in sublist]

    valid_f1 = f1_score(flat_valid_labels, flat_valid_preds, average='macro') 

    LOGGER.info(f'F1 score: {valid_f1:.4f}')
    if valid_f1 > best_f1:
        LOGGER.info(f"Model improve: {best_f1:.4f} -> {valid_f1:.4f}")
        torch.save(model.state_dict(), f'{CFG.only_model_name}_epoch_{epoch}.pth')
        best_f1 = valid_f1
