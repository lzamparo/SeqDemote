import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from load_pytorch import DNase_Train_Dataset, DNase_Valid_Dataset

data_path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap.h5")
save_dir = "basset_repro_pytorch"

batch_size = 128
momentum = 0.98
clipping = 7
cuda = True

learning_rate_schedule = {
0: 0.003,
10: 0.002,
20: 0.0001}

validate_every = 1
save_every = 1

train_loss = nn.BCELoss()
valid_loss = nn.BCELoss()
train_dataset = DNase_Train_Dataset(data_path, transform=None)
valid_dataset = DNase_Valid_Dataset(data_path, transform=None)
data_cast = lambda x: torch.autograd.Variable(x).float()
label_cast = lambda y: torch.autograd.Variable(y).float()


class BassetRepro(nn.Module):
    
    def __init__(self, input_size=(4,1,600)):
        super(BassetRepro, self).__init__()
        self.conv1 = nn.Conv2d(4, 300, kernel_size=(1,19))
        self.bn1 = nn.BatchNorm2d(300)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        
        self.conv2 = nn.Conv2d(300, 200, kernel_size=(1,11))
        self.bn2 = nn.BatchNorm2d(200)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        
        self.conv3 = nn.Conv2d(200, 200, kernel_size=(1,7))
        self.bn3 = nn.BatchNorm2d(200)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        
        conv_size = self._get_conv_output(input_size)
        
        self.fc1 = nn.Linear(conv_size, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(1000,1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(1000,164)
        self.out_layer = nn.Sigmoid()
        
        
    def forward(self, input):
        
        x = self.pool1(F.relu(self.bn1(self.conv1(input))))
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # flatten layer
        x = x.view(x.size(0), -1)
        
        x = self.drop1(self.bn4(self.fc1(x)))
        
        x = self.drop2(self.bn5(self.fc2(x)))
        
        x = self.fc3(x)
        
        return self.out_layer(x)
        
    
    # helper function to calculate number of units to expect for 
    # FC layers
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (1,3), stride=(1,3)))
        x = F.relu(F.max_pool2d(self.conv2(x), (1,4), stride=(1,4)))
        x = F.relu(F.max_pool2d(self.conv3(x), (1,4), stride=(1,4)))
        return x    
    
    
net = BassetRepro()


def init_weights(m, gain=nn.init.calculate_gain('relu')):
    ''' Recursively initalizes the weights of a network. '''
    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    
    if isinstance(m, nn.Conv2d):
        torch.nn.init.orthogonal(m.weight, gain)
        m.bias.data.fill_(0.1)

net.apply(init_weights)

weights, biases = [], []
for name, p in net.named_parameters():
    if 'bias' in name:
        biases += [p]
        
    else:
        weights += [p]
  
  
# Initialize the params        
optimizer = torch.optim.RMSprop