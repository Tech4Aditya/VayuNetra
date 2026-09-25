import torch
from torch import nn
class TrackGRU(nn.Module):
    def __init__(self,features=5,hidden=96,layers=2):
        super().__init__(); self.gru=nn.GRU(features,hidden,layers,batch_first=True,dropout=.15 if layers>1 else 0); self.head=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x):
        z,_=self.gru(x); return self.head(z[:,-1])
