from pathlib import Path
import pandas as pd,numpy as np,json
ROOT=Path(__file__).resolve().parents[2]; LABEL=ROOT/'data'/'labels'; OUT=ROOT/'data'/'processed'; OUT.mkdir(exist_ok=True)
def pick(df,names):
 low={str(c).lower().strip():c for c in df.columns}
 for n in names:
  if n in low:return low[n]
 for c in df.columns:
  s=str(c).lower().strip()
  if any(n in s for n in names):return c
 return None
def main():
 frames=[]
 for f in sorted(LABEL.glob('*_besttrack.csv')):
  d=pd.read_csv(f); lat=pick(d,['latitude','lat','lat_deg','center_lat','storm_lat']); lon=pick(d,['longitude','lon','long','lon_deg','center_lon','storm_lon']); storm=pick(d,['cyclone_id','storm_id','cyclone','storm','name','system_id']); ts=pick(d,['timestamp','datetime','date','time','valid_time','observation_time']); wind=pick(d,['wind_kt','wind','max_wind','maximum_wind','vmax']); pressure=pick(d,['pressure_hpa','pressure','mslp','min_pressure']); size=pick(d,['size_nmi','size_nm','size','radius_nmi','radius_nm'])
  if not all([lat,lon,storm,ts]): continue
  o=pd.DataFrame({'storm':d[storm].astype(str).str.strip(),'timestamp':d[ts].astype(str),'lat':pd.to_numeric(d[lat],errors='coerce'),'lon':pd.to_numeric(d[lon],errors='coerce'),'wind':pd.to_numeric(d[wind],errors='coerce') if wind else np.nan,'pressure':pd.to_numeric(d[pressure],errors='coerce') if pressure else np.nan,'size':pd.to_numeric(d[size],errors='coerce') if size else np.nan}).dropna(subset=['lat','lon'])
  if not o.empty: frames.append(o)
 if not frames: raise RuntimeError('No best-track files found.')
 d=pd.concat(frames,ignore_index=True).drop_duplicates(['storm','timestamp']); d['_time']=pd.to_datetime(d.timestamp,errors='coerce',utc=True,format='mixed'); d=d.dropna(subset=['_time']).sort_values(['storm','_time'])
 seq=[]
 for storm,g in d.groupby('storm'):
  g=g.reset_index(drop=True)
  for i in range(4,len(g)-3):
   h=g.iloc[i-4:i+1]; x=np.nan_to_num(h[['lat','lon','wind','pressure','size']].to_numpy(float),nan=0.0); cur=g.iloc[i]; ys=[]
   for k in (1,2,3):
    nxt=g.iloc[i+k]; ys.extend([nxt.lat-cur.lat,nxt.lon-cur.lon])
   seq.append({'storm':storm,'timestamp':cur.timestamp,'x':json.dumps(x.tolist()),'y':json.dumps(ys)})
 out=pd.DataFrame(seq)
 if out.empty: raise RuntimeError('No 3-horizon sequences could be built; need at least 8 observations per storm.')
 storms=sorted(out.storm.unique()); rng=np.random.default_rng(42); rng.shuffle(storms); n=len(storms); ntr=max(1,int(.6*n)); nva=max(1,int(.2*n)); ntr=min(ntr,max(1,n-2)); nva=min(nva,max(1,n-ntr-1)); tr=set(storms[:ntr]); va=set(storms[ntr:ntr+nva]); te=set(storms[ntr+nva:])
 for name,s in [('train',tr),('val',va),('test',te)]: out[out.storm.isin(s)].to_csv(OUT/f'track_multi_{name}.csv',index=False); print(name,len(out[out.storm.isin(s)]),'storms=',len(s))
 (OUT/'track_multi_split.json').write_text(json.dumps({'train':sorted(tr),'val':sorted(va),'test':sorted(te)},indent=2))
if __name__=='__main__':main()
