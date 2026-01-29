# MILOF-TCN

pip install matplot, torch, PyYaml, pandas

cd ev-anomaly

cd ev-anomaly

python scripts/prepare_data.py --config configs/base.yaml
python scripts/inject_anomalies.py --config configs/base.yaml
python scripts/run_baselines.py --config configs/base.yaml
python scripts/run_milof_tcn.py --config configs/base.yaml
python scripts/run_sweep.py --config configs/base.yaml

CopyRight by
-Jeong Jaeyun-
