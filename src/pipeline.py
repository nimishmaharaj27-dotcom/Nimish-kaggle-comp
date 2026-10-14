"""Data splitting, augmentation, models and training loop (PyTorch, CPU/GPU)."""
import json
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from config import CACHE_DIR, CLASSES, OUT_DIR, SEED

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet statistics
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cached():
    import pandas as pd
    X = np.load(CACHE_DIR / "X_train.npy", mmap_mode="r")
    y = np.load(CACHE_DIR / "y_train.npy")
    meta = pd.read_csv(CACHE_DIR / "meta_train.csv", dtype={"image_id": str})
    return X, y, meta


def make_split(y, md5, val_frac=0.1):
    """Stratified split; identical files (same md5) are kept on the same side (no leakage)."""
    skf = StratifiedGroupKFold(n_splits=int(round(1 / val_frac)), shuffle=True, random_state=SEED)
    tr, va = next(skf.split(np.zeros(len(y)), y, groups=md5))
    return tr, va


def to_tensor(xb_uint8, size):
    x = torch.from_numpy(np.ascontiguousarray(xb_uint8)).permute(0, 3, 1, 2).float().div_(255)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x


def normalise(x):
    return (x - MEAN) / STD


def augment(x, size):
    """Batch augmentation on [0,1] tensors: h-flip, random scale/translate/rotate, brightness/contrast."""
    B = x.shape[0]
    flip = (torch.rand(B) < 0.5).float() * 2 - 1
    scale = torch.empty(B).uniform_(0.85, 1.15)
    ang = torch.empty(B).uniform_(-15, 15) * np.pi / 180
    tx, ty = torch.empty(B).uniform_(-0.1, 0.1), torch.empty(B).uniform_(-0.1, 0.1)
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    b = torch.empty(B, 1, 1, 1).uniform_(-0.15, 0.15)
    c = torch.empty(B, 1, 1, 1).uniform_(0.85, 1.15)
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * c + m + b).clamp_(0, 1)


class SmallCNN(nn.Module):
    """Baseline custom CNN trained from scratch: 4 conv blocks (Conv-BN-ReLU x2 + MaxPool) + GAP + dropout."""
    def __init__(self, n_classes=8, widths=(32, 64, 128, 256), drop=0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, pretrained=True, drop=0.2):
    if name == "smallcnn":
        return SmallCNN()
    return timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES), drop_rate=drop)


@torch.no_grad()
def predict_logits(model, X, size, bs=512, tta=False):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = normalise(to_tensor(X[i:i + bs], size)).contiguous(memory_format=torch.channels_last)
        lg = model(x).float()
        if tta:  # average with horizontally flipped view
            lg = (lg.softmax(1) + model(x.flip(3)).float().softmax(1)) / 2
            lg = lg.log()
        out.append(lg)
    return torch.cat(out).numpy()


def train(name, size, epochs, bs=128, lr=2e-3, wd=0.05, smoothing=0.1, pretrained=True,
          tag=None, patience=4, max_train=None):
    """Train with AdamW + OneCycle LR, label smoothing, augmentation, best-checkpoint/early stopping."""
    set_seed()
    tag = tag or f"{name}_{size}"
    X, y, meta = load_cached()
    tr, va = make_split(y, meta["md5"].values)
    if max_train:
        tr = tr[:max_train]
    Xva = np.ascontiguousarray(X[np.sort(va)]); yva = y[np.sort(va)]
    Xtr = np.ascontiguousarray(X[tr]); ytr = y[tr]
    model = build_model(name, pretrained).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    hist, best, bad = [], 0.0, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        perm = np.random.permutation(len(tr)); tl = tc = n = 0
        for i in range(len(tr) // bs):
            idx = np.sort(perm[i * bs:(i + 1) * bs])
            x = normalise(augment(to_tensor(Xtr[idx], size), size)).contiguous(memory_format=torch.channels_last)
            t = torch.from_numpy(ytr[idx])
            lg = model(x)
            loss = F.cross_entropy(lg, t, label_smoothing=smoothing)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(idx); tc += (lg.argmax(1) == t).sum().item(); n += len(idx)
            if i % 50 == 0:
                print(f"  ep{ep+1} it{i}/{len(tr)//bs} loss {tl/n:.4f} acc {tc/n:.4f} {time.time()-t0:.0f}s", flush=True)
        lv = predict_logits(model, Xva, size)
        vloss = F.cross_entropy(torch.from_numpy(lv), torch.from_numpy(yva)).item()
        vacc = float((lv.argmax(1) == yva).mean())
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, train_acc=tc / n, val_loss=vloss, val_acc=vacc,
                         lr=sched.get_last_lr()[0], seconds=time.time() - t0))
        print(f"epoch {ep+1}/{epochs} train_loss {tl/n:.4f} train_acc {tc/n:.4f} "
              f"val_loss {vloss:.4f} val_acc {vacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(hist, open(OUT_DIR / f"history_{tag}.json", "w"), indent=1)
        if vacc > best:
            best, bad = vacc, 0
            torch.save(model.state_dict(), CACHE_DIR / f"{tag}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break
    return hist
