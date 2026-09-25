from pathlib import Path
import json, os
import pandas as pd
from datetime import timedelta

ROOT=Path(__file__).resolve().parents[2]; LABELS=ROOT/'data'/'labels'; CFG=ROOT/'ml'/'config'/'nio_storms.json'; OUT=ROOT/'data'/'mosdac_jobs'; OUT.mkdir(parents=True,exist_ok=True)

def main():
    cfg=json.loads(CFG.read_text()); pad=int(cfg.get('date_padding_days',1)); dsid=cfg['insat_dataset_id']; bbox=cfg['bounding_box']
    index=[]
    for s in cfg['target_storms']:
        p=LABELS/f'{s.lower()}_besttrack.csv'
        if not p.exists(): continue
        d=pd.read_csv(p); t=pd.to_datetime(d.timestamp,errors='coerce',utc=True).dropna()
        if t.empty: continue
        start=(t.min()-pd.Timedelta(days=pad)).strftime('%Y-%m-%d'); end=(t.max()+pd.Timedelta(days=pad)).strftime('%Y-%m-%d')
        job=OUT/s; job.mkdir(exist_ok=True)
        conf={'user_credentials':{'username':'${MOSDAC_USERNAME}','password':'${MOSDAC_PASSWORD}'},'search_parameters':{'datasetId':dsid,'startTime':start,'endTime':end,'count':'100','boundingBox':bbox,'gId':''},'download_settings':{'download_path':str((ROOT/'data'/'raw'/'mosdac'/s.lower()).resolve()),'organize_by_date':True,'skip_user_prompt':True,'generate_error_log':True,'error_log_path':str((job/'error.log').resolve())}}
        (job/'config.template.json').write_text(json.dumps(conf,indent=2),encoding='utf-8')
        index.append({'storm':s,'datasetId':dsid,'startTime':start,'endTime':end,'boundingBox':bbox,'template':str(job/'config.template.json')})
    (OUT/'jobs.json').write_text(json.dumps(index,indent=2),encoding='utf-8'); print(f'Generated {len(index)} MOSDAC job templates in {OUT}')
if __name__=='__main__':main()
