from pathlib import Path
import json,math
import numpy as np,pandas as pd,torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from models.track_gru import TrackGRU
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'/'processed'; CKPT=ROOT/'checkpoints'; OUT=DATA/'track_evaluation.json'
def hav(a,b,c,d):
 r=6371.; p1,p2=math.radians(a),math.radians(c); dp=math.radians(c-a); dl=math.radians(d-b); q=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2; return 2*r*math.asin(math.sqrt(q))
def main():
 ck=torch.load(CKPT/'track_gru_best.pt',map_location='cpu',weights_only=False); m=TrackGRU(); m.load_state_dict(ck['model_state_dict']); m.eval(); mean=np.array(ck['mean']); std=np.array(ck['std'])
 df=pd.read_csv(DATA/'track_test.csv');
 if len(df)==0: raise RuntimeError('track_test.csv is empty.')
 errs=[]; dists=[]
 with torch.no_grad():
  for _,r in df.iterrows():
   x=(np.array(json.loads(r.x),float)-mean)/std; pred=m(torch.tensor(x,dtype=torch.float32).unsqueeze(0)).numpy()[0]; errs.append([pred[0]-r.dlat,pred[1]-r.dlon]);
   hist=np.array(json.loads(r.x),float); lat,lon=hist[-1,:2]; true=(lat+r.dlat,lon+r.dlon); got=(lat+pred[0],lon+pred[1]); dists.append(hav(true[0],true[1],got[0],got[1]))
 e=np.array(errs); out={'samples':len(df),'latitude_mae_deg':float(np.mean(np.abs(e[:,0]))),'longitude_mae_deg':float(np.mean(np.abs(e[:,1]))),'position_rmse_deg':float(np.sqrt(np.mean(e**2))),'great_circle_distance_error_km':float(np.mean(dists))}; OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
