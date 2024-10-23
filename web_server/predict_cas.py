import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import re
import random
import gc
import pickle
from transformers import T5Tokenizer, T5EncoderModel
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import uvicorn
import io
from PIL import Image
import base64
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import FastAPI
from fastapi.responses import Response

parser = argparse.ArgumentParser(description='Predict Options')
parser.add_argument('--port', type=int, default=8040, help='Port')
parser.add_argument('--use_model', type=str, default='12-27-10-28-52', help='Name of model to use')
parser.add_argument('--use_epoch', type=int, default='2', help='Epoch of model to use')
parser.add_argument('--step', type=int, default=16, help='Prediction window step')
#parser.add_argument('--seq', type=str, required=True, help='Amino acid sequence to predict')
#parser.add_argument('--title', type=str, default='sequence', help='Title of plot')
args = parser.parse_args()

port = args.port
use_model = args.use_model
use_epoch = args.use_epoch
step = args.step

# 预定义的变量
project_path = '/home/fengchr/staff/Cas-detector'

# 设定随机数种子
torch.manual_seed(201)
torch.cuda.manual_seed(88)

# 模型超参数
window_size = 64

#tools
def load_model(run_header, epoch):
	state = torch.load(project_path + "/"+run_header+".epoch_"+str(epoch)+".model.pkl")
	tmp = MyModel()
	tmp.to(device)
	tmp.load_state_dict(state['state_dict'])
	tmp.eval()
	return(tmp)

device = torch.device('cuda:0')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(project_path + '/embedding/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained(project_path + '/embedding/prot_t5_xl_half_uniref50-enc').to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.half()
model = model.eval()

with open(project_path + '/data/target_encode_dict.pkl', 'rb') as file:
	target_encode_dict = pickle.load(file)

all_type_num = len(target_encode_dict)
cas_types = list(target_encode_dict.keys())[-1]

#定义模型
class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.pool = nn.MaxPool2d(2)
		self.conv1 = nn.Conv2d(1, 128, 5, padding=2)
		self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
		self.conv3 = nn.Conv2d(128, 64, 5, padding=2)
		self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
		self.fc1 = nn.Linear(64 * int(1024/4) * int(window_size/4), 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, all_type_num)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.sigmoid(self.fc3(x))
		x = self.fc4(x)
		return x

#加载模型
my_model = load_model(use_model, use_epoch)

def split_seq_fixed(seq,window_size,step=8):
	if len(seq) < window_size:
		return([])
	out=[seq[i:i+window_size] for i in range(0,len(seq)-window_size+1,step)]
	return(out)
def predict_seq_old(mymodel, seq, step=step, title='sequence'):
	try:
		mymodel.eval()
		with torch.no_grad():
			split_lst = split_seq_fixed(seq,window_size,step=step)
			seqs_batch = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in split_lst]
			ids = tokenizer(seqs_batch, add_special_tokens=True, padding="longest")
			input_ids = torch.tensor(ids['input_ids']).to(device)
			attention_mask = torch.tensor(ids['attention_mask']).to(device)
			embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
			emb = embedding_repr.last_hidden_state[:,:window_size].detach().unsqueeze(1).to(device,dtype=torch.float32) # shape (batch x window_size x 1024)
			#print(emb.shape)
			outputs = mymodel(emb)
			#print(outputs.shape)
			outputs = torch.nn.Softmax(dim=1)(outputs)
			values = np.array(outputs.view(-1).cpu())
			types = list(target_encode_dict.keys())*len(outputs)
			indices = [i+(window_size+1)/2 for i in list(range(0,len(seq)-window_size+1,step)) for _ in range(len(list(target_encode_dict.keys())))]
			plotdata = pd.DataFrame({"indices":indices,"types":types,"values":values})
			plt.figure(figsize=(15,8))
			custom_palette = sns.color_palette(["#FFFF00","#FF0000","#C0C0C0","#808000","#008000","#00FFFF","#FF00FF","#800080","#00FF00","#008080","#000000"])
			plot = sns.lineplot(data=plotdata,x="indices",y="values",hue="types",palette=custom_palette)
			plot.set_title(title)
			plot.set_xlabel('pos')
			plot.set_ylabel('score')
			plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Threshold')

			img = io.BytesIO()
			plt.savefig(img, format='png')
			img.seek(0)
			image_data = img.getvalue()

			max_indices = torch.argmax(outputs, dim=1)
			preds = [list(target_encode_dict.keys())[i] for i in np.array(max_indices.cpu())]
			#print(preds)
			print()
			counter = Counter(preds)
			for element, count in counter.most_common():
				print(f"{element}: {count}")
			return str(counter.most_common()), image_data
	except exception as e:
		with open("/home/fengchr/staff/Cas-detector/web_server/error.txt",'w') as f:
			f.write(str(e))
		exit(0)

def predict_seq(mymodel, seq, step=step, title='sequence', return_type='html'):
	mymodel.eval()
	with torch.no_grad():
		split_lst = split_seq_fixed(seq,window_size,step=step)
		if len(split_lst) == 0:
			print("\""+title+"\" is too short!")
			return
		seqs_batch = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in split_lst]
		ids = tokenizer(seqs_batch, add_special_tokens=True, padding="longest")
		input_ids = torch.tensor(ids['input_ids']).to(device)
		attention_mask = torch.tensor(ids['attention_mask']).to(device)
		embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
		emb = embedding_repr.last_hidden_state[:,:window_size].detach().unsqueeze(1).to(device,dtype=torch.float32) # shape (batch x window_size x 1024)
		outputs = mymodel(emb)
		outputs = torch.nn.Softmax(dim=1)(outputs)
		#清理gpu缓存
		torch.cuda.empty_cache()
		
		mycolors = ["#91F876","#F64A4A","#81FBFB","#FFA07A","#008000","#C0C0C0","#0280CE","#800080","#F0F01F","#FF3EEE","#000000"]
		#饼图
		type_score = np.sum(np.array(outputs.cpu()), axis=0)
		all_types = list(target_encode_dict.keys())
		plotdata2 = pd.DataFrame({"type":all_types,"type_score":type_score})
		fig2 = go.Figure(data=[go.Pie(labels=all_types, values=type_score, marker=dict(colors=mycolors), pull=[0]*(len(all_types)-1)+[0.2])])
		fig2.update_layout(title=title, title_x=0.5)
		
		#折线图
		values = np.array(outputs.view(-1).cpu())
		types = all_types*len(outputs)
		indices = [i+(window_size+1)/2 for i in list(range(0,len(seq)-window_size+1,step)) for _ in range(len(all_types))]
		annotations = ["["+str(int(indices[i]-(window_size-1)/2))+", "+str(int(indices[i]+(window_size-1)/2))+"]: "+str(types[i])+"("+str(round(values[i],3))+")" for i in range(len(types))]
		plotdata1 = pd.DataFrame({"indices":indices,"types":types,"values":values,"annotations":annotations})
		# plt.figure(figsize=(15,8))
		# custom_palette = sns.color_palette(mycolors)
		# plot = sns.lineplot(data=plotdata,x="indices",y="values",hue="types",palette=custom_palette)
		# plot.set_title(title)
		# plot.set_xlabel('pos')
		# plot.set_ylabel('score')
		# plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Threshold')
		grouped_data = plotdata1.groupby('types')
		sorted_types = [all_types[i] for i in sorted(range(len(all_types)), key=lambda i: type_score[i], reverse=True)]
		fig1 = go.Figure()
		fig1.add_shape(
			type='line',
			x0=0,  # 设置水平线的起点 x 坐标
			y0=0.5,  # 设置水平线的 y 坐标
			x1=len(seq),  # 设置水平线的终点 x 坐标
			y1=0.5,  # 设置水平线的 y 坐标
			line=dict(color='#F4D03F', width=2, dash='dash')  # 设置线条的颜色、宽度和样式为虚线
		)
		fig1.add_shape(
			type='line',
			x0=0,  # 设置水平线的起点 x 坐标
			y0=0.8,  # 设置水平线的 y 坐标
			x1=len(seq),  # 设置水平线的终点 x 坐标
			y1=0.8,  # 设置水平线的 y 坐标
			line=dict(color='red', width=2, dash='dash')  # 设置线条的颜色、宽度和样式为虚线
		)
		for group in sorted_types:
			data = grouped_data.get_group(group)
			fig1.add_trace(go.Scatter(x=data['indices'], y=data['values'], mode='lines+markers',line=dict(color=mycolors[all_types.index(group)]), name=group, text=data['annotations'], textposition='top center', hoverinfo='text'))
		fig1.update_layout(title=title, title_x=0.5, xaxis_title='pos', yaxis_title='score', xaxis=dict(range=[0, len(seq)]), yaxis=dict(range=[0-0.05, 1+0.05]))
		
		#柱形图
		
		
		#预测结果统计
		max_indices = torch.argmax(outputs, dim=1)
		preds = [all_types[i] for i in np.array(max_indices.cpu())]
		#print(preds)
		print()
		counter = Counter(preds)
		for element, count in counter.most_common():
			print(f"{element}: {count}")
		
		fig1.show()
		fig2.show()
		fig1_html = pio.to_html(fig1)
		fig2_html = pio.to_html(fig2)
		with open('/home/fengchr/staff/Cas-detector/web_server/log.txt','a') as f:
			f.write("1")
		if return_type == 'html':
			return str(counter.most_common()), fig1_html, fig2_html
		elif return_type == 'raw':
			return str(counter.most_common()), sorted_types, plotdata1, plotdata2


app = FastAPI()
@app.get("/predict/")
def predict(seq: str, title: str='sequence', step: str=str(step), return_type: str='html'):
	with open('/home/fengchr/staff/Cas-detector/web_server/log.txt','a') as f:
		f.write("0")
	step=int(step)
	if return_type == 'html':
		res, fig1_html, fig2_html = predict_seq(my_model, seq, step=step, title=title, return_type=return_type)
		# img_base64 = base64.b64encode(img_io).decode('utf-8')
		with open('/home/fengchr/staff/Cas-detector/web_server/log.txt','a') as f:
			f.write("2")
		return{
			"statistics": res,
			"fig1_html": fig1_html,
			"fig2_html": fig2_html
		}
		#return Response(img_io, media_type="image/png")
	elif return_type == 'raw':
		res, sorted_list, plotdata1, plodata2 = predict_seq(my_model, seq, step=step, title=title, return_type=return_type)
		# img_base64 = base64.b64encode(img_io).decode('utf-8')
		with open('/home/fengchr/staff/Cas-detector/web_server/log.txt','a') as f:
			f.write("3")
		return{
			"statistics": res,
			"sorted_list": sorted_list,
			"plotdata1": plotdata1.to_dict(orient='dict'),
			"plodata2": plodata2.to_dict(orient='dict')
		}

if __name__ == "__main__":
   uvicorn.run("predict_cas:app", host="127.0.0.1", port=port, reload=True)
