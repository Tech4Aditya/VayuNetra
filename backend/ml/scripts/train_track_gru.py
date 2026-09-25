from pathlib import Path
import argparse,json,random
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from models.track_gru import TrackGRU
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'/'processed'; CKPT=ROOT/'checkpoints'
class DS(Dataset):
 def __init__(self,df): self.x=[np.array(json.loads(v),dtype='float32') for v in df.x]; self.y=df[['dlat','dlon']].to_numpy('float32')
 def __len__(self): return len(self.x)
 def __getitem__(self,i): return torch.tensor(self.x[i]),torch.tensor(self.y[i])
def main(a):
 random.seed(42); np.random.seed(42); torch.manual_seed(42); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 tr=pd.read_csv(DATA/'track_train.csv'); va=pd.read_csv(DATA/'track_val.csv');
 if len(tr)<10: raise RuntimeError('Track training set too small. Need multiple storm histories with lat/lon.')
 if len(va)<1:
  # Tiny datasets can have no dedicated validation storm. Keep the model
  # trainable rather than failing; evaluation remains on the held-out test set.
  va=tr.sample(min(max(1,len(tr)//10),len(tr)),random_state=42).copy()
 # Fit normalization only on train.
 arr=np.concatenate([np.array(json.loads(v),dtype='float32') for v in tr.x]); mean=arr.mean(0); std=arr.std(0)+1e-6
 def norm(df):
  d=df.copy(); d.x=[json.dumps(((np.array(json.loads(v))-mean)/std).tolist()) for v in d.x]; return d
 tr,va=norm(tr),norm(va); dl=DataLoader(DS(tr),a.batch,shuffle=True); vl=DataLoader(DS(va),a.batch)
 m=TrackGRU().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4); lossfn=nn.SmoothL1Loss(); best=1e99
 for ep in range(1,a.epochs+1):
  m.train(); tl=0
  for x,y in dl: x,y=x.to(dev),y.to(dev); o=m(x); l=lossfn(o,y); opt.zero_grad(); l.backward(); opt.step(); tl+=l.item()*len(x)
  m.eval(); vloss=0
  with torch.no_grad():
   for x,y in vl: vloss+=lossfn(m(x.to(dev)),y.to(dev)).item()*len(x)
  vloss/=len(va); print(f'epoch {ep:03d} train={tl/len(tr):.6f} val={vloss:.6f}')
  if vloss<best:
   best=vloss; CKPT.mkdir(exist_ok=True); torch.save({'model_state_dict':m.state_dict(),'input_features':5,'hidden':96,'layers':2,'mean':mean,'std':std,'target':'delta_lat_delta_lon','interval_hours':3,'epoch':ep},CKPT/'track_gru_best.pt')
 print('saved',CKPT/'track_gru_best.pt')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=80); p.add_argument('--batch',type=int,default=32); p.add_argument('--lr',type=float,default=2e-3); main(p.parse_args())
