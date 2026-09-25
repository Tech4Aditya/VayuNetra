import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, c1,c2):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(c1,c2,3,padding=1),nn.BatchNorm2d(c2),nn.ReLU(),nn.MaxPool2d(2))
    def forward(self,x): return self.net(x)

class INSATMultiTask(nn.Module):
    def __init__(self, in_channels=2, n_classes=7, predict_size=False):
        super().__init__(); self.predict_size=predict_size
        self.features=nn.Sequential(ConvBlock(in_channels,32),ConvBlock(32,64),ConvBlock(64,128),ConvBlock(128,192))
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(192,128),nn.ReLU(),nn.Dropout(.15))
        self.category=nn.Linear(128,n_classes)
        self.wind=nn.Linear(128,1); self.pressure=nn.Linear(128,1)
        self.size=nn.Linear(128,1) if predict_size else None
    def forward(self,x):
        z=self.fc(self.pool(self.features(x)))
        out={'category':self.category(z),'wind':self.wind(z).squeeze(-1),'pressure':self.pressure(z).squeeze(-1)}
        if self.size is not None: out['size']=self.size(z).squeeze(-1)
        return out
