$ErrorActionPreference = 'Stop'
Write-Host '=== VayuNetra ML V2 ==='
Write-Host '[1/5] Preparing INSAT storm-level splits...'
python backend/ml/scripts/prepare_insat_splits.py
Write-Host '[2/5] Training normalized + class-weighted INSAT model...'
python backend/ml/scripts/train_insat_v2.py --epochs 100 --patience 15
Write-Host '[3/5] Evaluating INSAT on held-out storms...'
python backend/ml/scripts/evaluate_insat_v2.py
Write-Host '[4/5] Preparing multi-horizon track sequences...'
python backend/ml/scripts/prepare_track_multi.py
Write-Host '[5/5] Training/evaluating 3h/6h/9h track GRU...'
python backend/ml/scripts/train_track_multi.py --epochs 100 --patience 15
python backend/ml/scripts/evaluate_track_multi.py
Write-Host '=== VayuNetra ML V2 COMPLETE ==='
