from pathlib import Path
import argparse,json,random
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.insat_complete import INSATMultiTask

ROOT=Path(__file__).resolve().parents[2]
SPLIT=ROOT/'data'/'processed'/'insat_splits'; CKPT=ROOT/'checkpoints'; OUT=ROOT/'data'/'processed'
CATS=['Depression','Deep Depression','Cyclonic Storm','Severe Cyclonic Storm','Very Severe Cyclonic Storm','Extremely Severe Cyclonic Storm','Super Cyclonic Storm']

def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def finite_num(v):
    try:
        x=float(v); return x if np.isfinite(x) else np.nan
    except Exception: return np.nan

class DS(Dataset):
    def __init__(self,df,stats): self.df=df.reset_index(drop=True); self.stats=stats
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        a=np.squeeze(np.load(r.tir_path)).astype('float32'); b=np.squeeze(np.load(r.wv_path)).astype('float32')
        if a.shape!=(128,128) or b.shape!=(128,128): raise ValueError(f'Bad shape {a.shape} {b.shape}')
        a=(a-self.stats['tir_mean'])/self.stats['tir_std']; b=(b-self.stats['wv_mean'])/self.stats['wv_std']
        x=torch.from_numpy(np.stack([a,b]))
        y={'category':torch.tensor(int(r.category),dtype=torch.long),
           'wind':torch.tensor((float(r.wind_kt)-self.stats['wind_mean'])/self.stats['wind_std'],dtype=torch.float32),
           'pressure':torch.tensor((float(r.pressure_hpa)-self.stats['pressure_mean'])/self.stats['pressure_std'],dtype=torch.float32)}
        if 'size_nmi' in self.df.columns:
            sv=finite_num(r.get('size_nmi',np.nan))
            if np.isfinite(sv): y['size']=torch.tensor((sv-self.stats['size_mean'])/self.stats['size_std'],dtype=torch.float32)
        return x,y

def image_stats(df):
    # Sample pixels to keep RAM low.
    tir=[]; wv=[]
    for _,r in df.iterrows():
        a=np.asarray(np.load(r.tir_path),dtype='float32').reshape(-1); b=np.asarray(np.load(r.wv_path),dtype='float32').reshape(-1)
        step=max(1,len(a)//4096); tir.append(a[::step][:4096]); step=max(1,len(b)//4096); wv.append(b[::step][:4096])
    tir=np.concatenate(tir); wv=np.concatenate(wv)
    return float(tir.mean()),float(tir.std()+1e-6),float(wv.mean()),float(wv.std()+1e-6)

def stats_from_train(df):
    tm,ts,wm,ws=image_stats(df)
    wind=pd.to_numeric(df.wind_kt,errors='coerce').to_numpy(float); pressure=pd.to_numeric(df.pressure_hpa,errors='coerce').to_numpy(float)
    out={'tir_mean':tm,'tir_std':ts,'wv_mean':wm,'wv_std':ws,
         'wind_mean':float(np.nanmean(wind)),'wind_std':float(np.nanstd(wind)+1e-6),
         'pressure_mean':float(np.nanmean(pressure)),'pressure_std':float(np.nanstd(pressure)+1e-6)}
    if 'size_nmi' in df.columns:
        size=pd.to_numeric(df.size_nmi,errors='coerce').to_numpy(float); out['size_mean']=float(np.nanmean(size)); out['size_std']=float(np.nanstd(size)+1e-6)
    return out

def class_weights(df,n=7):
    c=df.category.astype(int).value_counts().to_dict(); total=len(df); w=np.ones(n,dtype='float32')
    for i in range(n): w[i]=total/max(1,n*c.get(i,0))
    w=w*(n/w.sum()); return torch.tensor(w)

def eval_val(model,loader,stats,device,ce,hub,size_ok):
    model.eval(); total=0; ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); o=model(x)
            l=ce(o['category'],y['category'].to(device)) + .25*hub(o['wind'],y['wind'].to(device)) + .25*hub(o['pressure'],y['pressure'].to(device))
            if size_ok and 'size' in o and 'size' in y: l += .10*hub(o['size'],y['size'].to(device))
            total += l.item()*len(x); ys.extend(y['category'].numpy().tolist()); ps.extend(o['category'].argmax(1).cpu().numpy().tolist())
    return total/max(1,len(loader.dataset)), accuracy_score(ys,ps), f1_score(ys,ps,average='macro',zero_division=0)

def main(a):
    seed(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device)
    tr=pd.read_csv(SPLIT/'insat_train.csv'); va=pd.read_csv(SPLIT/'insat_val.csv')
    if len(tr)<10: raise RuntimeError('INSAT training split is too small.')
    stats=stats_from_train(tr); size_ok='size_nmi' in tr.columns and pd.to_numeric(tr.size_nmi,errors='coerce').notna().sum()>5
    dl=DataLoader(DS(tr,stats),a.batch,shuffle=True); vl=DataLoader(DS(va,stats),a.batch,shuffle=False)
    model=INSATMultiTask(predict_size=size_ok).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5)
    ce=nn.CrossEntropyLoss(weight=class_weights(tr).to(device)); hub=nn.SmoothL1Loss()
    best=float('inf'); bad=0
    for ep in range(1,a.epochs+1):
        model.train(); total=0
        for x,y in dl:
            x=x.to(device); o=model(x)
            loss=ce(o['category'],y['category'].to(device)) + .25*hub(o['wind'],y['wind'].to(device)) + .25*hub(o['pressure'],y['pressure'].to(device))
            if size_ok and 'size' in o and 'size' in y: loss += .10*hub(o['size'],y['size'].to(device))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(x)
        v,acc,f1=eval_val(model,vl,stats,device,ce,hub,size_ok); sched.step(v)
        print(f'epoch {ep:03d} train={total/len(tr):.4f} val={v:.4f} val_acc={acc:.3f} val_macro_f1={f1:.3f} lr={opt.param_groups[0]["lr"]:.2e}')
        if v < best-1e-5:
            best=v; bad=0; CKPT.mkdir(exist_ok=True)
            torch.save({'model_state_dict':model.state_dict(),'category_names':CATS,'in_channels':2,'predict_size':size_ok,'stats':stats,'class_weights':class_weights(tr).tolist(),'epoch':ep,'loss_definition':'CE(weighted)+normalized regression SmoothL1'},CKPT/'insat_v2_best.pt')
        else: bad+=1
        if bad>=a.patience: print('early stopping'); break
    (OUT/'insat_v2_stats.json').write_text(json.dumps(stats,indent=2)); print('saved',CKPT/'insat_v2_best.pt')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch',type=int,default=8); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--patience',type=int,default=15); p.add_argument('--seed',type=int,default=42); main(p.parse_args())
