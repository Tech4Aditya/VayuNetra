from pathlib import Path
import argparse,json,random,numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from models.track_gru_multi import TrackGRUMulti
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'/'processed'; CKPT=ROOT/'checkpoints'
class DS(Dataset):
 def __init__(self,df,mean,std): self.x=[np.array(json.loads(v),dtype='float32') for v in df.x]; self.y=[np.array(json.loads(v),dtype='float32').reshape(3,2) for v in df.y]; self.mean=mean; self.std=std
 def __len__(self):return len(self.x)
 def __getitem__(self,i):
  x=((self.x[i]-self.mean)/self.std).astype(np.float32)
  y=self.y[i].astype(np.float32)
  return torch.from_numpy(x),torch.from_numpy(y)
def main(a):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tr=pd.read_csv(DATA/'track_multi_train.csv');va=pd.read_csv(DATA/'track_multi_val.csv');
 if len(tr)<20:raise RuntimeError('Track multi training set too small.')
 arr=np.concatenate([np.array(json.loads(v),float) for v in tr.x]);mean=arr.mean(0).astype(np.float32);std=(arr.std(0)+1e-6).astype(np.float32); dl=DataLoader(DS(tr,mean,std),a.batch,shuffle=True);vl=DataLoader(DS(va,mean,std),a.batch);m=TrackGRUMulti().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);lossfn=nn.SmoothL1Loss();best=1e99;bad=0
 for ep in range(1,a.epochs+1):
  m.train();tot=0
  for x,y in dl:x,y=x.to(dev),y.to(dev);o=m(x);l=lossfn(o,y);opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();tot+=l.item()*len(x)
  m.eval();v=0
  with torch.no_grad():
   for x,y in vl:v+=lossfn(m(x.to(dev)),y.to(dev)).item()*len(x)
  v/=max(1,len(va));print(f'epoch {ep:03d} train={tot/len(tr):.6f} val={v:.6f}')
  if v<best-1e-6:best=v;bad=0;CKPT.mkdir(exist_ok=True);torch.save({'model_state_dict':m.state_dict(),'mean':mean,'std':std,'input_features':5,'horizons':[3,6,9],'epoch':ep},CKPT/'track_gru_multi_best.pt')
  else:bad+=1
  if bad>=a.patience:print('early stopping');break
 print('saved',CKPT/'track_gru_multi_best.pt')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--patience',type=int,default=15);main(p.parse_args())
