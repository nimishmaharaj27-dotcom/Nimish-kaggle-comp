"""Decode every image once, resize to IMG_SIZE x IMG_SIZE and store as uint8 arrays.

Also records original sizes and an MD5 of the raw file bytes (for duplicate detection).
Preprocessing here is deterministic and identical for train and test images.
"""
import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from config import CACHE_DIR, CLASS_TO_IDX, DATA_DIR, IMG_SIZE


def _load(path):
    raw = open(path, "rb").read()
    with Image.open(path) as im:
        w, h = im.size
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8), w, h, hashlib.md5(raw).hexdigest()


def build(split, ids):
    paths = [DATA_DIR / split / split / f"{i}.jpg" for i in ids]
    with Pool() as p:
        res = p.map(_load, paths, chunksize=256)
    X = np.stack([r[0] for r in res])
    meta = pd.DataFrame({"image_id": ids, "w": [r[1] for r in res],
                         "h": [r[2] for r in res], "md5": [r[3] for r in res]})
    np.save(CACHE_DIR / f"X_{split}.npy", X)
    meta.to_csv(CACHE_DIR / f"meta_{split}.csv", index=False)
    return X, meta


if __name__ == "__main__":
    lab = pd.read_csv(DATA_DIR / "train_labels.csv", dtype=str)
    build("train", lab["image_id"].tolist())
    np.save(CACHE_DIR / "y_train.npy", lab["class"].map(CLASS_TO_IDX).values.astype(np.int64))
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype=str)
    # test files are zero-padded (00001.jpg) while the submission uses plain ints (1)
    test_ids = [f"{int(i):05d}" for i in sub["image_id"]]
    build("test", test_ids)
    print("cached", CACHE_DIR)
