from pathlib import Path
import pandas as pd, numpy as np, json, re
ROOT=Path(__file__).resolve().parents[2]; LABEL=ROOT/'data'/'labels'; OUT=ROOT/'data'/'processed'; OUT.mkdir(exist_ok=True)

def pick(df, names):
    low={str(c).lower().strip():c for c in df.columns}
    for n in names:
        if n in low:return low[n]
    for c in df.columns:
        s=str(c).lower().strip()
        if any(n in s for n in names): return c
    return None

def load_sources():
    # Prefer dedicated best-track files: unlike the temporal inference
    # manifests, these contain the full chronological storm track.
    candidates=[]
    besttracks=sorted(LABEL.glob('*_besttrack.csv'))
    if besttracks:
        candidates.extend(besttracks)
    # Then fall back to the full TCIR manifest. Temporal manifests are
    # inference windows and may contain only a handful of rows per storm.
    candidates.append(LABEL/'tcir_manifest.csv')
    candidates.extend([LABEL/'tcir_temporal_train.csv', LABEL/'tcir_temporal_val.csv', LABEL/'tcir_temporal_test.csv'])
    candidates=[f for f in candidates if f.exists()]
    # If best-track files exist, use only those for training. They are full
    # chronological tracks; mixing them with temporal windows can duplicate
    # or truncate sequences.
    if besttracks:
        candidates=besttracks

    frames=[]
    for f in candidates:
        try:
            d=pd.read_csv(f)
            lat=pick(d,['latitude','lat','lat_deg','center_lat','storm_lat'])
            lon=pick(d,['longitude','lon','long','lon_deg','center_lon','storm_lon'])
            storm=pick(d,['cyclone_id','storm_id','cyclone','storm','name','system_id'])
            ts=pick(d,['timestamp','datetime','date','time','valid_time','observation_time'])
            wind=pick(d,['wind_kt','wind','max_wind','maximum_wind','vmax'])
            pressure=pick(d,['pressure_hpa','pressure','mslp','min_pressure'])
            size=pick(d,['size_nmi','size_nm','size','radius_nmi','radius_nm'])
            if not all([lat,lon,storm,ts]):
                print('skip',f,'missing required columns')
                continue
            out=pd.DataFrame({
                'storm':d[storm].astype(str).str.strip(),
                'timestamp':d[ts].astype(str),
                'lat':pd.to_numeric(d[lat],errors='coerce'),
                'lon':pd.to_numeric(d[lon],errors='coerce')
            })
            out['wind']=pd.to_numeric(d[wind],errors='coerce') if wind else np.nan
            out['pressure']=pd.to_numeric(d[pressure],errors='coerce') if pressure else np.nan
            out['size']=pd.to_numeric(d[size],errors='coerce') if size else np.nan
            out=out.dropna(subset=['lat','lon'])
            if not out.empty:
                print('loaded',f.name,'rows=',len(out),'storms=',out.storm.nunique())
                frames.append(out)
        except Exception as e:
            print('skip',f,e)

    if not frames:
        raise RuntimeError('No track source with storm/timestamp/lat/lon was found.')

    d=pd.concat(frames,ignore_index=True)

    d=d.drop_duplicates(['storm','timestamp']).copy()
    d['_time']=pd.to_datetime(d['timestamp'],errors='coerce',utc=True,format='mixed')
    d=d.dropna(subset=['_time']).sort_values(['storm','_time']).drop(columns=['_time'])
    # Remove storms with fewer than six chronological observations.
    counts=d.groupby('storm').size()
    valid=counts[counts>=6].index
    d=d[d.storm.isin(valid)].copy()
    print('usable storms=',d.storm.nunique(),'rows=',len(d))
    return d

def main():
    d=load_sources(); seq=[]
    for storm,g in d.groupby('storm'):
        g=g.reset_index(drop=True)
        # Use five observed positions as history and predict the NEXT
        # displacement. The previous version accidentally used the final
        # transition inside the input window as the target (data leakage).
        for i in range(4,len(g)-1):
            h=g.iloc[i-4:i+1]
            x=h[['lat','lon','wind','pressure','size']].to_numpy(float); x=np.nan_to_num(x,nan=0.0)
            nxt=g.iloc[i+1]
            cur=g.iloc[i]
            target=np.array([nxt.lat-cur.lat,nxt.lon-cur.lon],dtype=float)
            seq.append({'storm':storm,'timestamp':cur.timestamp,'x':json.dumps(x.tolist()),'dlat':target[0],'dlon':target[1]})
    out=pd.DataFrame(seq)
    if out.empty: raise RuntimeError('No track sequences could be built. Need at least 6 chronological observations per cyclone.')
    # Deterministic storm-level split with no storm appearing in multiple sets.
    storms=sorted(out.storm.unique(), key=str)
    rng=np.random.default_rng(42); rng.shuffle(storms)
    n=len(storms); ntr=max(1,int(.6*n)); nva=max(1,int(.2*n))
    if n>=3:
        ntr=min(ntr,n-2); nva=min(nva,n-ntr-1)
    tr=set(storms[:ntr]); va=set(storms[ntr:ntr+nva]); te=set(storms[ntr+nva:])
    if not te: te={storms[-1]}; tr=set(storms[:-1]); va=set()
    for name,s in [('train',tr),('val',va),('test',te)]:
        part=out[out.storm.isin(s)]
        part.to_csv(OUT/f'track_{name}.csv',index=False)
        print(name,len(part),'storms=',len(s))
    (OUT/'track_storm_split.json').write_text(json.dumps({'train':sorted(tr),'val':sorted(va),'test':sorted(te)},indent=2))
if __name__=='__main__': main()
