from pathlib import Path
import json, numpy as np, pandas as pd, torch, sys
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from models.insat_complete import INSATMultiTask
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'/'processed'; CKPT=ROOT/'checkpoints'
def main():
 ck=torch.load(CKPT/'insat_complete_best.pt',map_location='cpu',weights_only=False); m=INSATMultiTask(predict_size=bool(ck.get('predict_size',False))); m.load_state_dict(ck['model_state_dict']); m.eval(); st=ck['stats']; test_csv = DATA/'insat_test.csv'
 if not test_csv.exists(): test_csv = DATA/'insat_splits'/'insat_test.csv'
 df=pd.read_csv(test_csv);
 if len(df)==0: raise RuntimeError('Empty INSAT test split.')
 ys=[]; ps=[]
 for _,r in df.iterrows():
  a=np.squeeze(np.load(r.tir_path)).astype('float32'); b=np.squeeze(np.load(r.wv_path)).astype('float32'); x=np.stack([(a-st['tir_mean'])/st['tir_std'],(b-st['wv_mean'])/st['wv_std']]);
  with torch.no_grad(): o=m(torch.tensor(x).unsqueeze(0));
  ys.append([int(r.category),float(r.wind_kt),float(r.pressure_hpa),float(r.size_nmi) if 'size_nmi' in df.columns and str(r.get('size_nmi','')) not in ('','nan','None') else np.nan]); ps.append([int(o['category'].argmax()),float(o['wind'][0]),float(o['pressure'][0]),float(o['size'][0]) if 'size' in o else np.nan])
 y=np.array(ys,float); p=np.array(ps,float); out={'samples':len(df),'category_accuracy':float(accuracy_score(y[:,0],p[:,0])),'category_macro_f1':float(f1_score(y[:,0],p[:,0],average='macro',zero_division=0)),'wind_mae_kt':float(mean_absolute_error(y[:,1],p[:,1])),'wind_rmse_kt':float(np.sqrt(mean_squared_error(y[:,1],p[:,1]))),'wind_r2':float(r2_score(y[:,1],p[:,1])),'pressure_mae_hpa':float(mean_absolute_error(y[:,2],p[:,2])),'pressure_rmse_hpa':float(np.sqrt(mean_squared_error(y[:,2],p[:,2]))),'pressure_r2':float(r2_score(y[:,2],p[:,2]))}
 if np.isfinite(y[:,3]).sum()>0 and 'size' in ck and False: pass
 (DATA/'insat_complete_evaluation.json').write_text(json.dumps(out,indent=2)); pd.DataFrame([out]).to_csv(DATA/'insat_complete_evaluation.csv',index=False); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
