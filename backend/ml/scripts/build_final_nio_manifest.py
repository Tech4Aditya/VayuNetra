from pathlib import Path
import pandas as pd, json
ROOT=Path(__file__).resolve().parents[2]; LABELS=ROOT/'data'/'labels'; OUT=ROOT/'data'/'processed'; CFG=ROOT/'ml'/'config'/'nio_storms.json'

def main():
    cfg=json.loads(CFG.read_text()); frames=[]
    for s in cfg['target_storms']:
        p=LABELS/f'{s.lower()}_insat_manifest.csv'
        if not p.exists(): continue
        d=pd.read_csv(p); d['storm']=s; frames.append(d)
    if not frames: raise SystemExit('No INSAT manifests found.')
    out=pd.concat(frames,ignore_index=True); out.to_csv(OUT/'nio_insat_manifest_final.csv',index=False)
    summary=out.groupby('storm').size().sort_values(ascending=False).to_dict(); (OUT/'nio_dataset_summary.json').write_text(json.dumps({'storms':summary,'total_rows':len(out)},indent=2),encoding='utf-8')
    print(json.dumps({'storms':summary,'total_rows':len(out)},indent=2))
if __name__=='__main__':main()
