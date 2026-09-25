from pathlib import Path
import json, os

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CKPT=ROOT/'checkpoints'; OUT=DATA/'processed'/'final_system_evaluation.json'

def exists(p): return p.exists()
def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None

def main():
    checks={
      'tcir_temporal_checkpoint': exists(CKPT/'tcir_temporal_best.pt'),
      'tcir_intensity_checkpoint': exists(CKPT/'tcir_intensity_best.pt'),
      'insat_v2_checkpoint': exists(CKPT/'insat_v2_best.pt'),
      'track_gru_multi_checkpoint': exists(CKPT/'track_gru_multi_best.pt'),
      'insat_test_split': exists(DATA/'processed'/'insat_splits'/'insat_test.csv'),
      'track_test_split': exists(DATA/'processed'/'track_multi_test.csv'),
      'track_evaluation': exists(DATA/'processed'/'track_multi_evaluation.json'),
      'insat_evaluation': exists(DATA/'processed'/'insat_v2_evaluation.json'),
      'insat_baseline': exists(DATA/'processed'/'insat_baseline_evaluation.json'),
      'insat_failure_analysis': exists(DATA/'processed'/'insat_failure_analysis.json'),
    }
    out={'status':'complete' if all(checks.values()) else 'incomplete','checks':checks,'evaluations':{}}
    for name,f in [('track',DATA/'processed'/'track_multi_evaluation.json'),('insat',DATA/'processed'/'insat_v2_evaluation.json'),('insat_baseline',DATA/'processed'/'insat_baseline_evaluation.json'),('insat_failures',DATA/'processed'/'insat_failure_analysis.json')]:
        obj=load_json(f)
        if obj is not None: out['evaluations'][name]=obj
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
