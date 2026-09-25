import torch
from torch import nn
class TrackGRUMulti(nn.Module):
    def __init__(self,in_features=5,hidden=96,horizons=3):
        super().__init__(); self.gru=nn.GRU(in_features,hidden,num_layers=2,batch_first=True,dropout=.15); self.head=nn.Sequential(nn.Linear(hidden,96),nn.ReLU(),nn.Dropout(.1),nn.Linear(96,horizons*2)); self.horizons=horizons
    def forward(self,x):
        z,_=self.gru(x); return self.head(z[:,-1]).view(-1,self.horizons,2)
