"""Tasks 5/6: validation metrics, confusion matrix, error analysis and Kaggle submission file."""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import CACHE_DIR, CLASSES, DATA_DIR, OUT_DIR
from pipeline import build_model, load_cached, make_split, predict_logits

p = argparse.ArgumentParser()
p.add_argument("--model", default="resnet18"); p.add_argument("--size", type=int, default=64)
p.add_argument("--tag", default=None); p.add_argument("--submit", action="store_true")
a = p.parse_args()
tag = a.tag or f"{a.model}_{a.size}"

X, y, meta = load_cached()
_, va = make_split(y, meta["md5"].values); va = np.sort(va)
model = build_model(a.model, pretrained=False).to(memory_format=torch.channels_last)
model.load_state_dict(torch.load(CACHE_DIR / f"{tag}.pt")); model.eval()

Xva, yva = np.ascontiguousarray(X[va]), y[va]
res = {}
for tta in (False, True):
    lg = predict_logits(model, Xva, a.size, tta=tta)
    pr = lg.argmax(1)
    res["tta" if tta else "plain"] = dict(
        acc=float((pr == yva).mean()),
        loss=float(torch.nn.functional.nll_loss(torch.from_numpy(lg).log_softmax(1), torch.from_numpy(yva)))
        if not tta else float(torch.nn.functional.nll_loss(torch.from_numpy(lg), torch.from_numpy(yva))))
    if tta:
        np.save(CACHE_DIR / f"val_logits_{tag}.npy", lg)
        rep = classification_report(yva, pr, target_names=CLASSES, output_dict=True, digits=4)
        pd.DataFrame(rep).T.round(4).to_csv(OUT_DIR / f"classification_report_{tag}.csv")
        print(classification_report(yva, pr, target_names=CLASSES, digits=4))
        cm = confusion_matrix(yva, pr)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[0], cbar=False)
        sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".3f", cmap="Blues", xticklabels=CLASSES,
                    yticklabels=CLASSES, ax=ax[1], cbar=False)
        ax[0].set_title("Confusion matrix (counts)"); ax[1].set_title("Row-normalised (recall)")
        for x in ax: x.set_xlabel("predicted"); x.set_ylabel("true"); x.tick_params(axis="x", rotation=45)
        plt.tight_layout(); plt.savefig(OUT_DIR / f"confusion_{tag}.png", dpi=150); plt.close()
        wrong = np.where(pr != yva)[0]
        conf = torch.from_numpy(lg).exp().max(1).values.numpy()
        wrong = wrong[np.argsort(-conf[wrong])][:24]
        fig, axes = plt.subplots(3, 8, figsize=(14, 6))
        for ax_, i in zip(axes.ravel(), wrong):
            ax_.imshow(X[va[i]]); ax_.axis("off")
            ax_.set_title(f"T:{CLASSES[yva[i]]}\nP:{CLASSES[pr[i]]}", fontsize=7)
        plt.suptitle("Most confident misclassifications (validation)"); plt.tight_layout()
        plt.savefig(OUT_DIR / f"errors_{tag}.png", dpi=130); plt.close()
json.dump(res, open(OUT_DIR / f"val_metrics_{tag}.json", "w"), indent=1); print(res)

if a.submit:
    Xt = np.load(CACHE_DIR / "X_test.npy", mmap_mode="r")
    lt = predict_logits(model, Xt, a.size, tta=True)
    np.save(CACHE_DIR / f"test_logits_{tag}.npy", lt)
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype=str)
    sub["class"] = [CLASSES[i] for i in lt.argmax(1)]
    sub.to_csv(OUT_DIR / f"submission_{tag}.csv", index=False)
    print(sub["class"].value_counts()); print("wrote", OUT_DIR / f"submission_{tag}.csv")
