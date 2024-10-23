import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
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
ori = h5py.File(project_path + '/data/data_12_16.h5', 'r')
cas_emb = ori['cas_emb']
cas_labels = ori['cas_labels']
with h5py.File(project_path + '/data/data_cas_12_26.h5', 'w') as f:
    f.create_dataset('cas_emb', data=cas_emb)
    f.create_dataset('cas_labels', data=cas_labels)
ori.close()
