import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

import struct
from copy import deepcopy
from time import time, sleep
import gc
import copy

gpu_boole = torch.cuda.is_available()

class MLP(nn.Module):
    def __init__(self, input_size, width=500, num_classes=10):
        super(MLP, self).__init__()

        self.ff1 = nn.Linear(input_size, width)

        self.ff2 = nn.Linear(width, width)
        self.ff3 = nn.Linear(width, width)
        self.ff4 = nn.Linear(width, width)
        self.ff5 = nn.Linear(width, width)

        self.ff_out = nn.Linear(width, num_classes)
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        
        ##BN:
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)
        self.bn4 = nn.BatchNorm1d(width)
        self.bn5 = nn.BatchNorm1d(width)
        
    def forward(self, input_data):

        out = self.relu(self.ff1(input_data))

        out = self.relu(self.ff2(out))
        out = self.relu(self.ff3(out))
        out = self.relu(self.ff4(out))
        out = self.relu(self.ff5(out))

        out = self.ff_out(out)

        return out

