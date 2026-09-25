from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'processed'
SPLIT = DATA / 'insat_splits' / 'insat_test.csv'
OUT = DATA / 'insat_baseline_evaluation.json'

CATS = ['Depression','Deep Depression','Cyclonic Storm','Severe Cyclonic Storm','Very Severe Cyclonic Storm','Extremely Severe Cyclonic Storm','Super Cyclonic Storm']

def metric(y, p):
    return {
        'mae': float(mean_absolute_error(y, p)),
        'rmse': float(np.sqrt(mean_squared_error(y, p))),
        'r2': float(r2_score(y, p)) if len(np.unique(y)) > 1 else None,
    }

def main():
    df = pd.read_csv(SPLIT)
    if df.empty:
        raise RuntimeError('INSAT test split is empty.')
    ycat = df['category'].astype(int).to_numpy()
    counts = pd.Series(ycat).value_counts().sort_index()
    majority = int(counts.idxmax())
    pcat = np.full(len(df), majority, dtype=int)
    wind = pd.to_numeric(df['wind_kt'], errors='coerce').to_numpy(float)
    pressure = pd.to_numeric(df['pressure_hpa'], errors='coerce').to_numpy(float)
    pwind = np.full(len(df), np.nanmean(pd.read_csv(DATA/'insat_splits'/'insat_train.csv')['wind_kt']), dtype=float)
    ppressure = np.full(len(df), np.nanmean(pd.read_csv(DATA/'insat_splits'/'insat_train.csv')['pressure_hpa']), dtype=float)
    out = {
        'samples': int(len(df)),
        'baseline': 'majority-class classification + training-set mean regression',
        'majority_class_index': majority,
        'majority_class': CATS[majority],
        'category_accuracy': float(accuracy_score(ycat, pcat)),
        'category_macro_f1': float(f1_score(ycat, pcat, average='macro', zero_division=0)),
        'test_class_counts': {CATS[int(k)]: int(v) for k,v in counts.items()},
        'wind': metric(wind, pwind),
        'pressure': metric(pressure, ppressure),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__': main()
