"""Shared configuration for the Facial Feature Classification pipeline."""
import os
from pathlib import Path

SEED = 42
IMG_SIZE = 96  # cached/model input resolution (source crops are ~40-100 px)
CLASSES = ["CatEye", "CatFace", "CatMouth", "CatNose",
           "DogEye", "DogFace", "DogMouth", "DogNose"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

ROOT = Path(__file__).resolve().parents[1]
# Folder containing train/, test/, train_labels.csv, sample_submission.csv
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ROOT / "cache"))
OUT_DIR = ROOT / "outputs"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True, parents=True)
