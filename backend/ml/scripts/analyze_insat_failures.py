from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.insat_complete import INSATMultiTask

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'processed'
CKPT = ROOT / 'checkpoints' / 'insat_v2_best.pt'
TEST = DATA / 'insat_splits' / 'insat_test.csv'
OUT = DATA / 'insat_failure_analysis.json'
CATS = ['Depression','Deep Depression','Cyclonic Storm','Severe Cyclonic Storm','Very Severe Cyclonic Storm','Extremely Severe Cyclonic Storm','Super Cyclonic Storm']

def main():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = INSATMultiTask(predict_size=bool(ck.get('predict_size', False)))
    model.load_state_dict(ck['model_state_dict']); model.eval()
    st = ck['stats']; df = pd.read_csv(TEST)
    y=[]; p=[]; rows=[]
    for idx,r in df.iterrows():
        a=np.squeeze(np.load(r.tir_path)).astype('float32'); b=np.squeeze(np.load(r.wv_path)).astype('float32')
        x=np.stack([(a-st['tir_mean'])/st['tir_std'],(b-st['wv_mean'])/st['wv_std']])
        with torch.no_grad(): o=model(torch.tensor(x).unsqueeze(0))
        pred_cat=int(o['category'].argmax(1).item())
        pred_w=float(o['wind'].item())*st['wind_std']+st['wind_mean']
        pred_p=float(o['pressure'].item())*st['pressure_std']+st['pressure_mean']
        true_cat=int(r.category); true_w=float(r.wind_kt); true_p=float(r.pressure_hpa)
        y.append(true_cat); p.append(pred_cat)
        rows.append({'index':int(idx),'true_category':CATS[true_cat],'predicted_category':CATS[pred_cat], 'wind_abs_error_kt':abs(true_w-pred_w), 'pressure_abs_error_hpa':abs(true_p-pred_p)})
    cm=confusion_matrix(y,p,labels=list(range(7)))
    hardest=sorted(rows,key=lambda z:z['wind_abs_error_kt']+z['pressure_abs_error_hpa'],reverse=True)[:10]
    out={'samples':len(df),'collapse_observation': 'Model prediction distribution is heavily concentrated in one or more classes; inspect confusion matrix before claiming robust 7-class classification.','confusion_matrix':cm.tolist(),'predicted_class_counts':{CATS[i]:int(p.count(i)) for i in range(7)},'hardest_regression_cases':hardest}
    OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
