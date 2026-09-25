from pathlib import Path
import json, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / 'data' / 'labels'
CFG = ROOT / 'ml' / 'config' / 'nio_storms.json'
OUT = LABELS / 'nio_besttracks'
OUT.mkdir(parents=True, exist_ok=True)


def norm(x):
    return str(x).strip().upper()


def main():
    try:
        import imdtrack as imd
    except ImportError:
        raise SystemExit('imdtrack is required. Run: python -m pip install imdtrack')

    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    targets = [norm(x) for x in cfg['target_storms']]
    print('Loading current IMD RSMC New Delhi best-track record...')
    bt = imd.load(update=True)
    storms = bt.storms.copy()
    storms['name_norm'] = storms['name'].fillna('').map(norm)
    obs = bt.observations.copy()
    obs['name_norm'] = obs['name'].fillna('').map(norm)

    found = {}
    for target in targets:
        srow = storms[storms['name_norm'] == target]
        if srow.empty:
            print(f'[MISSING] {target}')
            continue
        storm_id = str(srow.iloc[0]['storm_id'])
        d = obs[obs['storm_id'].astype(str) == storm_id].copy()
        if d.empty:
            print(f'[MISSING OBS] {target} ({storm_id})')
            continue
        d['time'] = pd.to_datetime(d['time'], errors='coerce', utc=True)
        out = pd.DataFrame({
            'storm': target,
            'timestamp': d['time'].astype(str),
            'lat': pd.to_numeric(d['lat'], errors='coerce'),
            'lon': pd.to_numeric(d['lon'], errors='coerce'),
            'wind_kt': pd.to_numeric(d['wind'], errors='coerce'),
            'pressure_hpa': pd.to_numeric(d['pressure'], errors='coerce'),
            'category': d['grade'].astype(str),
            'ci_no': pd.to_numeric(d['ci_no'], errors='coerce'),
            'pressure_drop_hpa': pd.to_numeric(d['pressure_drop'], errors='coerce'),
            'oci_diameter_deg': pd.to_numeric(d['oci_diameter'], errors='coerce'),
        })
        out = out.dropna(subset=['lat','lon']).sort_values('timestamp').drop_duplicates('timestamp')
        out.to_csv(LABELS / f'{target.lower()}_besttrack.csv', index=False)
        out.to_csv(OUT / f'{target.lower()}_besttrack.csv', index=False)
        found[target] = {'storm_id': storm_id, 'rows': len(out), 'start': out['timestamp'].min(), 'end': out['timestamp'].max()}
        print(f'[OK] {target}: {len(out)} observations ({storm_id})')

    missing = [x for x in targets if x not in found]
    summary = {'requested': targets, 'found': found, 'missing': missing, 'count_found': len(found), 'count_requested': len(targets)}
    (OUT / 'nio_besttrack_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if missing:
        raise SystemExit(f'Best-track collection incomplete: missing {missing}')
    print(f'COMPLETE: {len(found)}/{len(targets)} target storms exported.')

if __name__ == '__main__': main()
