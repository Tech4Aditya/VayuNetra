from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT/'data'/'labels'
PROCESSED = ROOT/'data'/'processed'
OUT = PROCESSED/'insat_splits'
CFG = Path(__file__).resolve().parents[1]/'config'/'splits.json'
OUT.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP={'D':0,'Depression':0,'DD':1,'Deep Depression':1,'CS':2,'Cyclonic Storm':2,'SCS':3,'Severe Cyclonic Storm':3,'VSCS':4,'Very Severe Cyclonic Storm':4,'ESCS':5,'Extremely Severe Cyclonic Storm':5,'SuCS':6,'SUCS':6,'Super Cyclonic Storm':6}

def process_path(storm, file_value):
    p=Path(file_value)
    sample=p.stem
    return PROCESSED/f'insat_{storm.lower()}_calibrated'/sample/'tir1_radiance.npy', PROCESSED/f'insat_{storm.lower()}_calibrated'/sample/'wv_radiance.npy'

def main():
    cfg=json.loads(CFG.read_text())
    assigned={s.upper():k for k,v in cfg.items() if k in ('train','val','test') for s in v}
    rows={k:[] for k in ('train','val','test')}
    for mf in sorted(LABELS.glob('*_insat_manifest.csv')):
        storm=mf.stem.replace('_insat_manifest','').upper()
        if storm not in assigned:
            continue
        split=assigned[storm]
        with mf.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                try:
                    tir,wv=process_path(storm,r['file'])
                    if not tir.exists() or not wv.exists(): continue
                    cat=str(r.get('category','')).strip()
                    if cat not in CATEGORY_MAP: continue
                    out={'storm':storm,'timestamp':r.get('insat_timestamp',''),'tir_path':str(tir),'wv_path':str(wv),'category':CATEGORY_MAP[cat],'category_name':cat,'wind_kt':r.get('wind_kt',''),'pressure_hpa':r.get('pressure_hpa','')}
                    for key in ('size_nmi','size_nm','size','radius_nmi','storm_size_nmi'):
                        if r.get(key,'') not in ('',None): out['size_nmi']=r[key]; break
                    rows[split].append(out)
                except Exception:
                    continue
    for split,data in rows.items():
        path=OUT/f'insat_{split}.csv'
        fields=sorted({k for r in data for k in r})
        if not fields: fields=['storm','timestamp','tir_path','wv_path','category','category_name','wind_kt','pressure_hpa']
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
        print(f'{split}: {len(data)} -> {path}')
    print('Storm-level split complete.')

if __name__=='__main__': main()
