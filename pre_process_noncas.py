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

project_path = '/home/fengchr/staff/Cas-detector'
window_size = 64  # 片段长度

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
def split_seq(seq,window_size,step):#具体step为多少需要外部判断
	if len(seq) < window_size:
		return([])
	out=[seq[i:i+window_size] for i in range(random.randint(0,step-1),len(seq)-window_size+1,step)]
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
all_type_num = len(cas_types)+1

#用T5编码
target_encode_dict = {}
for i in range(all_type_num):
	target_encode_dict[(cas_types+["nonCas"])[i]] = i

for mm in range(11,16):
	nonCas_seqs = read_fasta(project_path + '/data/nonCas_split'+str(mm)+'.fasta')
	noncas_set=[(i,"nonCas",nonCas_seqs[i]) for i in nonCas_seqs]

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
	print("7")
	# with open(project_path + '/data/noncas_data.pkl', 'wb') as file:
	#	 pickle.dump(noncas_set1, file)

	with h5py.File(project_path + '/data/data_noncas.'+str(mm)+'.h5', 'w') as f:
		f.create_dataset('noncas_emb', data=noncas_emb)
		f.create_dataset('noncas_labels', data=noncas_labels)
	del noncas_emb, noncas_labels, split_lst
	gc.collect()
	#编码为模型可用形式
	# encoded_cas_seq_lst = [encode_aa_lst(split_seq(i[2],window_size)) for i in cas_set]#每个元素是一个seq的encoded kmer list
	# cas_set1 = [(j,torch.tensor(target_encode_dict[cas_set[i][1]])) for i in range(len(cas_set)) for j in encoded_cas_seq_lst[i]]
	# cas_size = len(cas_set1)
	# random.shuffle(cas_set1)

	# encoded_noncas_seq_lst = [encode_aa_lst(split_seq(i[2],window_size)) for i in noncas_set]#每个元素是一个seq的encoded kmer list
	# noncas_set1 = [(j,torch.tensor(target_encode_dict["nonCas"])) for i in encoded_noncas_seq_lst for j in i]
	# noncas_size = len(noncas_set1)
	# random.shuffle(noncas_set1)


	print("12")
