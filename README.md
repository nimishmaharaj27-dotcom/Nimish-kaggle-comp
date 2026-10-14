# ENGE707 Assessment 3 – Facial Feature Classification Challenge

Eight-class cat/dog facial-feature classifier (Kaggle: ENGE707-2026-S2). Pretrained ResNet-18 fine-tuned on 64×64 crops,
99.25% validation accuracy (99.37% with flip TTA).

| File | Purpose |
|---|---|
| `ENGE707_Facial_Feature_Classification.ipynb` | Documented pipeline with all results |
| `src/` | `cache_data.py`, `eda.py`, `pipeline.py`, `train.py`, `evaluate_predict.py` |
| `run_experiments.sh` | Trains ResNet-18 and the SmallCNN baseline |
| `submission.csv` | Final Kaggle submission file |
| `outputs/` | Figures, metrics, training histories |
| `report/` | Written report (markdown + Word) |
| `evidence/` | Kaggle leaderboard screenshot (add after submitting) |

## Reproduce
```bash
pip install torch torchvision timm scikit-learn pandas matplotlib seaborn pillow
export DATA_DIR=/path/to/unzipped/data   # train/, test/, train_labels.csv, sample_submission.csv
export CACHE_DIR=/path/to/cache
python src/cache_data.py && python src/eda.py
bash run_experiments.sh
cd src && python evaluate_predict.py --model resnet18 --size 64 --submit
```
Trained on a 16-core CPU (~9 min/epoch). Seed 42.
