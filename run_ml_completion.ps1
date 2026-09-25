$ErrorActionPreference = 'Stop'
Write-Host '=== VayuNetra ML Completion ===' -ForegroundColor Cyan
Write-Host '[1/4] Preparing storm-level INSAT splits...'
python backend/ml/scripts/prepare_insat_splits.py
Write-Host '[2/4] Training INSAT model...'
python backend/ml/scripts/train_insat_complete.py --epochs 50
Write-Host '[3/4] Preparing track sequences...'
python backend/ml/scripts/prepare_track_dataset.py
Write-Host '[4/4] Training learned GRU track predictor...'
python backend/ml/scripts/train_track_gru.py --epochs 80
Write-Host 'Done. Run evaluations:' -ForegroundColor Green
Write-Host 'python backend/ml/scripts/evaluate_insat_complete.py'
Write-Host 'python backend/ml/scripts/evaluate_track.py'
