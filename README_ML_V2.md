# VayuNetra ML V2 completion pack

This is an incremental ML experiment pack. It does not replace the existing TCIR checkpoints. It adds a corrected INSAT training formulation and a multi-horizon track model.

Run `run_ml_v2.ps1` from the repository root after copying the pack's `backend/ml` files. New checkpoints use `insat_v2_best.pt` and `track_gru_multi_best.pt`, so the currently working demo remains intact until the new metrics are verified.
