#!/usr/local/anaconda/bin/python
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import re
import math
import random
import time
from datetime import datetime
import copy
import glob
import gc
import pickle
import h5py

# 预定义的变量
project_path = '/home/fengchr/staff/Cas-detector'

# 设定随机数种子
torch.manual_seed(201)
torch.cuda.manual_seed(88)

# 模型超参数
noncas_weight = 0.25 #nonCas对Cas样本的权重比
window_size = 64  # 片段长度
window_step = 25
lr = 0.01
momentum = 0.8
dropout = 0
batch_size = 128
lr_epoch_decay = 0.8  # 每个epoch的lr衰减
inspect_loss_decay = 0.5  # 每当loss下降为原来的inspect_loss_decay，便更新一次lr
lr_adapt_decay = 0.8  # 更新lr为原来的lr_adapt_decay
inspect_num = 3  # 以inspect_num个batch为单位判定模型收敛程度，要么1要么2

#tools
def read_fasta(input):
    with open(input,'r') as f:
        fasta = {}
        for line in f:
            line = line.strip()
            if line[0] == '>':
                header = line[1:]
            else:
                sequence = line
                fasta[header] = fasta.get(header,'') + sequence
    return fasta
def split_seq(seq,window_size,step=window_step):#具体step为多少需要外部判断
    if len(seq) < window_size:
        return([])
    out=[seq[i:i+window_size] for i in range(random.randint(0,step-1),len(seq)-window_size+1,step)]
    return(out)
def split_seq_rand(seq,window_size,num):#具体num为多少需要外部判断
    if len(seq) < window_size:
        return([])
    out=[]
    for i in range(num):
        tmp=random.randint(0,len(seq)-window_size)
        out.append(seq[tmp:tmp+window_size])
    return(out)
def one_hot_encode(lst):
    encode_dict = {}
    for i in range(len(lst)):
        tmp=np.zeros(len(lst))
        tmp[i]=1
        encode_dict[lst[i]]=tmp
    return encode_dict
encode_dict=one_hot_encode(["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","-"])
print("encode_dict:")
print(encode_dict)
def encode_aa_lst(aa_lst):
    return([torch.tensor(np.array([encode_dict.get(j,encode_dict["-"]) for j in list(aa)]), dtype=torch.float32) for aa in aa_lst])

from transformers import T5Tokenizer, T5EncoderModel

device = torch.device('cuda:0')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(project_path + '/embedding/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained(project_path + '/embedding/prot_t5_xl_half_uniref50-enc').to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.half()
model = model.eval()

#读取数据集
cas_files = glob.glob(project_path + '/data/merged_*.cdhit90.fa')
cas_types = [i.split("/")[-1][7:-11] for i in cas_files]

cas_set=[]
for i in cas_types:
    Cas_seqs = read_fasta(project_path + '/data/merged_'+i+'.cdhit90.fa')
    cas_set+=[(ii,i,Cas_seqs[ii]) for ii in Cas_seqs]
nonCas_seqs = read_fasta(project_path + '/data/nonCas.sample0.1.fa')
noncas_set=[(i,"nonCas",nonCas_seqs[i]) for i in nonCas_seqs]
print("1")
#整理为[(id, cas_type, seq)]
all_type_num = len(cas_types)+1

#用T5编码
target_encode_dict = {}
for i in range(all_type_num):
    target_encode_dict[(cas_types+["nonCas"])[i]] = i
with open(project_path + '/data/target_encode_dict.pkl', 'wb') as file:
    pickle.dump(target_encode_dict, file)

split_lst = [(cas_set[i][1],j) for i in range(len(cas_set)) for j in split_seq(cas_set[i][2],window_size)]
cas_emb = []
batch = []
print("2")
for i in range(len(split_lst)):
    batch.append(split_lst[i])
    if len(batch) >= 1000 or i >= len(split_lst)-1:
        print("start embedding batch "+str((i+1) // 1000))
        types_batch, seqs_batch = zip(*batch)
        seqs_batch = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs_batch]
        batch = []
        # tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer(seqs_batch, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        emb = embedding_repr.last_hidden_state[:,:window_size].detach().cpu().numpy() # shape (batch x window_size x 1024)
        cas_emb.append(emb)
        
        if i >= len(split_lst)-1:
            break
print("3")
cas_emb = np.concatenate(cas_emb,axis=0)
print("4")
cas_labels = np.array([target_encode_dict[i[0]] for i in split_lst])
cas_size = len(cas_labels)
print(str(cas_size))
with h5py.File(project_path + '/data/data_cas_12_22.h5', 'w') as f:
    f.create_dataset('cas_emb', data=cas_emb)
    f.create_dataset('cas_labels', data=cas_labels)

split_lst = [j for i in noncas_set for j in split_seq(i[2],window_size,step=window_size)]
print("5")
noncas_emb = []
batch = []
for i in range(len(split_lst)):
    batch.append(split_lst[i])
    if len(batch) >= 1000 or i >= len(split_lst)-1:
        print("start embedding batch "+str((i+1) // 1000))
        seqs_batch = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch]
        batch = []
        # tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer(seqs_batch, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        emb = embedding_repr.last_hidden_state[:,:window_size].detach().cpu().numpy() # shape (batch x window_size x 1024)
        noncas_emb.append(emb)
        
        if i >= len(split_lst)-1:
            break
print("6")
noncas_emb = np.concatenate(noncas_emb,axis=0)
noncas_labels = np.array([target_encode_dict["nonCas"] for i in range(noncas_emb.shape[0])])
noncas_size = len(noncas_labels)
print("7")
# with open(project_path + '/data/noncas_data.pkl', 'wb') as file:
#     pickle.dump(noncas_set1, file)

with h5py.File(project_path + '/data/data_noncas_12_22.h5', 'w') as f:
    f.create_dataset('noncas_emb', data=noncas_emb)
    f.create_dataset('noncas_labels', data=noncas_labels)
#编码为模型可用形式
# encoded_cas_seq_lst = [encode_aa_lst(split_seq(i[2],window_size)) for i in cas_set]#每个元素是一个seq的encoded kmer list
# cas_set1 = [(j,torch.tensor(target_encode_dict[cas_set[i][1]])) for i in range(len(cas_set)) for j in encoded_cas_seq_lst[i]]
# cas_size = len(cas_set1)
# random.shuffle(cas_set1)

# encoded_noncas_seq_lst = [encode_aa_lst(split_seq(i[2],window_size)) for i in noncas_set]#每个元素是一个seq的encoded kmer list
# noncas_set1 = [(j,torch.tensor(target_encode_dict["nonCas"])) for i in encoded_noncas_seq_lst for j in i]
# noncas_size = len(noncas_set1)
# random.shuffle(noncas_set1)

print("Cas items:\n"+str(len(cas_set))+"-"+str(cas_size))
print("Non-Cas items:\n"+str(len(noncas_set))+"-"+str(noncas_size))

print("12")
