import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


data_path = os.path.expanduser("~/projects/SeqDemote/data/ATAC/mouse_asa/mouse_asa_2k.h5")

### training params
batch_size = 128
subsequence_size = 200
momentum = 0.9
cuda=False

# set schedule for learning rate decreases
learning_rate_schedule = {
    0: 0.01,
    10: 0.001,
    20: 0.0001,
}

validate_every = 2
save_every = 5

# This is the DeepLIFT simulation arch for testing
#Convolution with 50 filters of width 11
#• ReLU
#• Convolution with 50 filters of width 11
#• ReLU
#• Global Average Pooling
#• Fully connected layer with 50 neurons
#• ReLU
#• Dropout layer with probability 0.5 of leaving out units
#• Fully connected layer with 3 neurons
#• Sigmoid output

class DeepLiftRegressor(nn.Module):
    
    def __init__(self):
        super(DeepLiftRegressor, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=50, kernel_size=(1,11))
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1,11))
        
        self.fc1 = nn.Linear(50, 50)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(50, 5)
        
    def forward(self, input):
        #l0 = nn.layers.InputLayer((batch_size, data_rows, 1, data_cols))
        
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        
        # Global pooling here
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = DeepLiftRegressor()



