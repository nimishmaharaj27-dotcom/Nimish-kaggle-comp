"""Task 1: dataset exploration. Saves figures/tables to outputs/."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CACHE_DIR, CLASSES, OUT_DIR, SEED

X = np.load(CACHE_DIR / "X_train.npy", mmap_mode="r")
y = np.load(CACHE_DIR / "y_train.npy")
mt = pd.read_csv(CACHE_DIR / "meta_train.csv", dtype={"image_id": str}); mt["y"] = y
ms = pd.read_csv(CACHE_DIR / "meta_test.csv", dtype={"image_id": str})

# class distribution
cnt = pd.Series(y).map(dict(enumerate(CLASSES))).value_counts().reindex(CLASSES)
fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.bar(CLASSES, cnt.values, color=["#4c72b0"] * 4 + ["#dd8452"] * 4)
ax.bar_label(bars); ax.set_ylabel("images"); ax.set_title("Training class distribution")
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(OUT_DIR / "class_distribution.png", dpi=150); plt.close()

# sample grid (raw cached 96px crops)
rng = np.random.RandomState(SEED)
fig, axes = plt.subplots(8, 8, figsize=(10, 10.5))
for r, c in enumerate(CLASSES):
    idx = rng.choice(np.where(y == r)[0], 8, replace=False)
    for k, i in enumerate(idx):
        axes[r, k].imshow(X[i]); axes[r, k].axis("off")
    axes[r, 0].set_title(c, loc="left", fontsize=9)
plt.tight_layout(); plt.savefig(OUT_DIR / "samples_per_class.png", dpi=120); plt.close()

# original sizes
mt["cls"] = mt["y"].map(dict(enumerate(CLASSES)))
mt["aspect"] = mt.w / mt.h
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
mt.boxplot(column="w", by="cls", ax=axs[0], rot=30); axs[0].set_yscale("log"); axs[0].set_title("Original width (px, log)")
mt.boxplot(column="aspect", by="cls", ax=axs[1], rot=30); axs[1].set_title("Aspect ratio w/h")
for a in axs: a.set_xlabel("")
plt.suptitle(""); plt.tight_layout(); plt.savefig(OUT_DIR / "size_by_class.png", dpi=150); plt.close()

# quality stats
ms["aspect"] = ms.w / ms.h
stats = {
    "n_train": len(mt), "n_test": len(ms),
    "train_dup_files": int(mt.md5.duplicated().sum()),
    "test_dup_files": int(ms.md5.duplicated().sum()),
    "test_in_train_dup": int(ms.md5.isin(mt.md5).sum()),
    "conflicting_label_dups": int((mt.groupby("md5").y.nunique() > 1).sum()),
    "train_min_side": int(min(mt.w.min(), mt.h.min())), "test_min_side": int(min(ms.w.min(), ms.h.min())),
    "train_max_side": int(max(mt.w.max(), mt.h.max())), "test_max_side": int(max(ms.w.max(), ms.h.max())),
    "train_nonsquare_frac": float((mt.w != mt.h).mean()), "test_nonsquare_frac": float((ms.w != ms.h).mean()),
    "train_tiny_lt24": int(((mt.w < 24) | (mt.h < 24)).sum()), "test_tiny_lt24": int(((ms.w < 24) | (ms.h < 24)).sum()),
}
# blur proxy: variance of Laplacian-like gradient on 96px grayscale (sample of 6000)
sel = np.sort(rng.choice(len(X), 6000, replace=False))
g = np.asarray(X[sel]).astype(np.float32).mean(-1)
lap = (g[:, 1:-1, 1:-1] * 4 - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])
blur = lap.reshape(len(sel), -1).var(1)
stats["blur_p5_laplace_var"] = float(np.percentile(blur, 5)); stats["blur_median_laplace_var"] = float(np.median(blur))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(blur, bins=60); ax[0].set_xlabel("Laplacian variance (sharpness)"); ax[0].set_title("Sharpness (6k sample)")
bl = sel[np.argsort(blur)[:8]]
ax[1].imshow(np.concatenate([X[i] for i in bl], 1)); ax[1].axis("off"); ax[1].set_title("8 blurriest in sample")
plt.tight_layout(); plt.savefig(OUT_DIR / "blur.png", dpi=150); plt.close()
mt.groupby("cls")[["w", "h"]].agg(["mean", "min", "max"]).round(1).to_csv(OUT_DIR / "size_stats.csv")
pd.Series(stats).to_json(OUT_DIR / "eda_stats.json", indent=1)
print(pd.Series(stats)); print(cnt)
