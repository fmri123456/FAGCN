# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
from layer import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.conv1 = GraphConvolution(input_dim,hidden_dim,False)
        self.conv2 = GraphConvolution(hidden_dim, hidden_dim,False)
        #两层卷积层
        self.fc1 = nn.Linear(161*hidden_dim,hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc3 = nn.Linear(hidden_dim // 4, num_classes)
        #三层全连接层
        # self.dropout = nn.Dropout(p=0.5)
    def forward(self,adjacency,input_feature,similarity):
        gcn1 = F.relu(self.conv1.forward(adjacency,input_feature,similarity))
        gcn2 = F.relu(self.conv2.forward(adjacency,gcn1,similarity))
        gc2_rl = gcn2.reshape(-1, 161*gcn2.shape[2])
        fc1 = F.relu(self.fc1(gc2_rl))
        fc2 = F.relu(self.fc2(fc1))
        # fc2 = F.softmax(fc1,dim=1)
        logits = self.fc3(fc2)
        return logits
