from pathlib import Path
import json,math,numpy as np,pandas as pd,torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from models.track_gru_multi import TrackGRUMulti
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data'/'processed';CKPT=ROOT/'checkpoints'
def hav(a,b,c,d):
 r=6371.;p1,p2=math.radians(a),math.radians(c);dp=math.radians(c-a);dl=math.radians(d-b);q=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2;return 2*r*math.asin(math.sqrt(max(0,q)))
def main():
 ck=torch.load(CKPT/'track_gru_multi_best.pt',map_location='cpu',weights_only=False);m=TrackGRUMulti();m.load_state_dict(ck['model_state_dict']);m.eval();mean=np.array(ck['mean']);std=np.array(ck['std']);df=pd.read_csv(DATA/'track_multi_test.csv');err=np.zeros((3,2));d=[[],[],[]]
 with torch.no_grad():
  for _,r in df.iterrows():
   h=np.array(json.loads(r.x),float);y=np.array(json.loads(r.y),float).reshape(3,2);pred=m(torch.tensor(((h-mean)/std),dtype=torch.float32).unsqueeze(0))[0].numpy();cur=h[-1,:2]
   for j in range(3):err[j]+=np.abs(pred[j]-y[j]);d[j].append(hav(cur[0]+y[j,0],cur[1]+y[j,1],cur[0]+pred[j,0],cur[1]+pred[j,1]))
 n=max(1,len(df));out={'samples':len(df),'horizons':{f'{3*(j+1)}h':{'latitude_mae_deg':float(err[j,0]/n),'longitude_mae_deg':float(err[j,1]/n),'great_circle_distance_error_km':float(np.mean(d[j]))} for j in range(3)}};(DATA/'track_multi_evaluation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
