# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 09:21:50 2020

@author: Gang Su
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
import sys


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, (1,3), (1,2), padding=(0,1))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 32, (2,3), (2,2), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.linear_c = nn.Linear(32*7*25, 4096)
        self.linear1 = nn.Linear(4116, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, output_size)
        self.linear4 = nn.Linear(4,10)
        self.dropout = nn.Dropout(p=0.2)
        # self.linear_c = nn.Linear(29, 2048)
        # self.linear1 = nn.Linear(2048, 2048)
        # self.linear2 = nn.Linear(2048, 1024)
        # self.linear3 = nn.Linear(1024, output_size)

    def forward(self, s,a,b):
        x = F.relu(self.bn1(self.conv1(s)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 32*7*25)
        x = self.dropout(F.relu(self.linear_c(x)))
        
        b = torch.tensor(b,dtype=torch.float).view(-1,4)
        b = F.relu(self.linear4(b))
        b = b.view(-1,10)
        a = torch.tensor(a,dtype=torch.float).view(-1,4)
        a = F.relu(self.linear4(a))
        a = a.view(-1,10)
        x = torch.cat([x,a,b],dim=1)
        x = x.view(-1,4116)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        # x = s.view(-1,29)
        # x = F.relu(self.linear_c(x))
        # for i in range(2):
        #  x = self.dropout(F.relu(self.linear1(x)))
        # x = self.dropout(F.relu(self.linear2(x)))
        # x = self.linear3(x)

        return x


class Train_Model():
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, tau):
        self._i_dim = (3,14,100) #input_dim
        self._o_dim = output_dim
        self.critic_lr = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)

        self.critic = Critic(self._i_dim, width, self._o_dim, num_layers)#.to(device)
        self.critic_target = Critic(self._i_dim, width, self._o_dim, num_layers)#.to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
            
         
    def critic_learn(self,s0,a0,r1,s1,gamma,done,a,b,c,d):
        y_pred = torch.zeros(self.batch_size,1)
        y_true = r1 + gamma * torch.max(self.critic_target(s1,b,d)) * done   # y_true

        # y_pred = self.critic(s0)[np.array(a0)]   # y_pred  
        a0 = torch.tensor(a0,dtype=torch.int64).reshape(-1,1)
        y_pred = torch.gather(self.critic(s0,a,c),1,a0)
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        # print("critic loss:",loss)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss
    
    def predict(self, s0,a,c):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.critic(s0,a,c).squeeze(0).detach()
        return a0
        
    def _update_model(self):
        self.critic_target = self.critic
        # for target_param, param  in zip(net_target.parameters(), net.parameters()):
            # target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                                       
    def soft_update(self,net_target, net):
        for target_param, param  in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = torch.load(model_file_path)
            self.critic = loaded_model
            print("load model_A successfully")
        else:
            sys.exit("Model number not found")
            
    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        torch.save(self.critic, os.path.join(path, 'trained_model.h5'))

class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self.critic = torch.load(os.path.join(model_path, 'trained_model.h5'))

    def predict_one(self, s0):
        """
        Predict the action values from a single state
        """
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.critic(s0).squeeze(0).detach().numpy()
        return a0
        