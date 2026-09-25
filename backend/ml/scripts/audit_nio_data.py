from pathlib import Path
import json, csv, hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT/'data'/'labels'; PROCESSED = ROOT/'data'/'processed'
CFG = ROOT/'ml'/'config'/'nio_storms.json'
OUT = PROCESSED/'nio_audit_report.json'; OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    cfg=json.loads(CFG.read_text()); storms=[s.upper() for s in cfg['target_storms']]
    report={'target_storms':storms,'besttrack':{},'insat_manifests':{},'calibrated_pairs':{},'orphan_files':[]}
    for s in storms:
        bt=LABELS/f'{s.lower()}_besttrack.csv'
        mf=LABELS/f'{s.lower()}_insat_manifest.csv'
        b={'exists':bt.exists()}; m={'exists':mf.exists()}
        if bt.exists():
            d=pd.read_csv(bt); b.update({'rows':len(d),'missing_lat':int(d.lat.isna().sum()),'missing_lon':int(d.lon.isna().sum()),'missing_time':int(pd.to_datetime(d.timestamp,errors='coerce').isna().sum()),'missing_wind':int(pd.to_numeric(d.wind_kt,errors='coerce').isna().sum()),'missing_pressure':int(pd.to_numeric(d.pressure_hpa,errors='coerce').isna().sum())})
        if mf.exists():
            d=pd.read_csv(mf); m['rows']=len(d); m['columns']=list(d.columns)
            missing_file=0; missing_tir=0; missing_wv=0
            for _,r in d.iterrows():
                f=str(r.get('file','')); sample=Path(f).stem
                base=PROCESSED/f'insat_{s.lower()}_calibrated'/sample
                if not base.exists(): missing_file+=1
                if not (base/'tir1_radiance.npy').exists(): missing_tir+=1
                if not (base/'wv_radiance.npy').exists(): missing_wv+=1
            m.update({'missing_sample_dirs':missing_file,'missing_tir':missing_tir,'missing_wv':missing_wv})
        report['besttrack'][s]=b; report['insat_manifests'][s]=m
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    bad=[]
    for s,v in report['besttrack'].items():
        if not v.get('exists') or v.get('rows',0)<8: bad.append(f'{s}: best-track')
    for s,v in report['insat_manifests'].items():
        if v.get('exists') and (v.get('missing_tir',0) or v.get('missing_wv',0)): bad.append(f'{s}: calibrated pairs incomplete')
    print('AUDIT_STATUS:', 'PASS' if not bad else 'ATTENTION -> '+', '.join(bad))
if __name__=='__main__':main()
