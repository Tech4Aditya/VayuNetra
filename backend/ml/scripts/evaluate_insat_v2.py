from pathlib import Path
import json,numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from models.insat_complete import INSATMultiTask
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'/'processed'; CKPT=ROOT/'checkpoints'
def metric(y,p): return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p))}
def main():
 ck=torch.load(CKPT/'insat_v2_best.pt',map_location='cpu',weights_only=False); m=INSATMultiTask(predict_size=bool(ck.get('predict_size',False))); m.load_state_dict(ck['model_state_dict']); m.eval(); st=ck['stats']
 path=DATA/'insat_splits'/'insat_test.csv'; df=pd.read_csv(path)
 yt=[]; pt=[]
 for _,r in df.iterrows():
  a=np.squeeze(np.load(r.tir_path)).astype('float32'); b=np.squeeze(np.load(r.wv_path)).astype('float32'); x=np.stack([(a-st['tir_mean'])/st['tir_std'],(b-st['wv_mean'])/st['wv_std']])
  with torch.no_grad(): o=m(torch.tensor(x).unsqueeze(0))
  pred=[int(o['category'].argmax()), float(o['wind'][0])*st['wind_std']+st['wind_mean'], float(o['pressure'][0])*st['pressure_std']+st['pressure_mean']]
  true=[int(r.category),float(r.wind_kt),float(r.pressure_hpa)]; yt.append(true); pt.append(pred)
 y=np.array(yt); p=np.array(pt); prec,rec,f1,_=precision_recall_fscore_support(y[:,0],p[:,0],labels=list(range(7)),zero_division=0)
 out={'samples':len(df),'category_accuracy':float(accuracy_score(y[:,0],p[:,0])),'category_macro_f1':float(f1_score(y[:,0],p[:,0],average='macro',zero_division=0)),'category_per_class':{str(i):{'precision':float(prec[i]),'recall':float(rec[i]),'f1':float(f1[i])} for i in range(7)},'confusion_matrix':confusion_matrix(y[:,0],p[:,0],labels=list(range(7))).tolist(),'wind':metric(y[:,1],p[:,1]),'pressure':metric(y[:,2],p[:,2])}
 out['category_names']=ck.get('category_names',[]); (DATA/'insat_v2_evaluation.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
