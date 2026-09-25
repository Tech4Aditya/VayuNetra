from pathlib import Path
import argparse,json,random
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.insat_complete import INSATMultiTask

ROOT=Path(__file__).resolve().parents[2]; SPLIT=ROOT/'data'/'processed'/'insat_splits'; CKPT=ROOT/'checkpoints'; OUT=ROOT/'data'/'processed'
CATS=['Depression','Deep Depression','Cyclonic Storm','Severe Cyclonic Storm','Very Severe Cyclonic Storm','Extremely Severe Cyclonic Storm','Super Cyclonic Storm']
def seed(s=42): random.seed(s); np.random.seed(s); torch.manual_seed(s)
class DS(Dataset):
 def __init__(self,df,stats=None): self.df=df.reset_index(drop=True); self.stats=stats
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]; a=np.squeeze(np.load(r.tir_path)).astype('float32'); b=np.squeeze(np.load(r.wv_path)).astype('float32')
  if a.shape!=(128,128) or b.shape!=(128,128): raise ValueError(f'Bad shape {a.shape} {b.shape}')
  if self.stats: a=(a-self.stats['tir_mean'])/self.stats['tir_std']; b=(b-self.stats['wv_mean'])/self.stats['wv_std']
  x=torch.from_numpy(np.stack([a,b])); y={'category':torch.tensor(int(r.category)),'wind':torch.tensor(float(r.wind_kt)),'pressure':torch.tensor(float(r.pressure_hpa))}
  if 'size_nmi' in self.df.columns and str(r.get('size_nmi','')) not in ('','nan','None'): y['size']=torch.tensor(float(r.size_nmi))
  return x,y

def calc_stats(df):
 vals={'tir':[],'wv':[]}
 for p in df.tir_path: vals['tir'].append(np.mean(np.load(p))); 
 for p in df.wv_path: vals['wv'].append(np.mean(np.load(p)))
 # Robust global statistics using per-image means/stds keeps memory low.
 tir=np.concatenate([np.asarray(np.load(p),dtype='float32').reshape(-1)[::128] for p in df.tir_path]); wv=np.concatenate([np.asarray(np.load(p),dtype='float32').reshape(-1)[::128] for p in df.wv_path])
 return {'tir_mean':float(tir.mean()),'tir_std':float(tir.std()+1e-6),'wv_mean':float(wv.mean()),'wv_std':float(wv.std()+1e-6)}

def run(args):
 seed(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device)
 tr=pd.read_csv(SPLIT/'insat_train.csv'); va=pd.read_csv(SPLIT/'insat_val.csv'); te=pd.read_csv(SPLIT/'insat_test.csv')
 if len(tr)==0: raise RuntimeError('No INSAT training samples. Run prepare_insat_splits.py and verify processed arrays.')
 stats=calc_stats(tr); size_ok='size_nmi' in tr.columns and tr.size_nmi.notna().sum()>0
 ds=DS(tr,stats); dv=DS(va,stats); dt=DS(te,stats); dl=DataLoader(ds,args.batch,shuffle=True); vl=DataLoader(dv,args.batch); tl=DataLoader(dt,args.batch)
 model=INSATMultiTask(predict_size=size_ok).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); hub=nn.SmoothL1Loss()
 best=1e99
 for ep in range(1,args.epochs+1):
  model.train(); total=0
  for x,y in dl:
   x=x.to(device); o=model(x); loss=ce(o['category'],y['category'].to(device))+0.03*hub(o['wind'],y['wind'].to(device))+0.01*hub(o['pressure'],y['pressure'].to(device))
   if size_ok: loss+=0.01*hub(o['size'],y['size'].to(device))
   opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(x)
  model.eval(); v=0
  with torch.no_grad():
   for x,y in vl:
    x=x.to(device); o=model(x); v+=ce(o['category'],y['category'].to(device)).item()*len(x)
  v/=max(1,len(va)); print(f'epoch {ep:03d} train={total/len(tr):.4f} val_cls={v:.4f}')
  if v<best:
   best=v; CKPT.mkdir(exist_ok=True); torch.save({'model_state_dict':model.state_dict(),'category_names':CATS,'in_channels':2,'predict_size':size_ok,'stats':stats,'epoch':ep},CKPT/'insat_complete_best.pt')
 (OUT/'insat_complete_stats.json').write_text(json.dumps(stats,indent=2)); print('saved',CKPT/'insat_complete_best.pt')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch',type=int,default=8); p.add_argument('--lr',type=float,default=1e-3); run(p.parse_args())
