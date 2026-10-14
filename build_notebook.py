"""Generates ENGE707_Facial_Feature_Classification.ipynb (run once, then execute the notebook)."""
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# ENGE707 Assessment 3 - Facial Feature Classification Challenge
**Kaggle username:** `<ADD YOUR KAGGLE USERNAME>`  |  **Student ID:** `<ADD>`

Eight-class classification (CatEye, CatFace, CatMouth, CatNose, DogEye, DogFace, DogMouth, DogNose) with a
pretrained **ResNet-18** fine-tuned end to end on CPU.

**How to reproduce**
1. `pip install torch torchvision timm scikit-learn pandas matplotlib seaborn`
2. Unzip the Kaggle data so `DATA_DIR` contains `train/train`, `test/test`, `train_labels.csv`, `sample_submission.csv`.
3. `export DATA_DIR=... CACHE_DIR=...` then run `python src/cache_data.py`, `python src/eda.py`,
   `bash run_experiments.sh`, `python src/evaluate_predict.py --model resnet18 --size 64 --submit`.
4. This notebook displays the saved results (set `RUN_TRAINING = True` to retrain from inside the notebook).""")

code("""import sys, os, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from config import *
RUN_TRAINING = False   # ~45 min per model on a 16-core CPU
if RUN_TRAINING:
    from pipeline import train
    train('resnet18', 64, 5, tag='resnet18_64')
    train('smallcnn', 64, 5, pretrained=False, lr=3e-3, tag='smallcnn_64')
print('device: CPU', '| image cache size:', IMG_SIZE)""")

md("## 1. Dataset exploration (Task 1)")
code("""stats = json.load(open(OUT_DIR/'eda_stats.json')); print(pd.Series(stats).to_string())
display(Image(str(OUT_DIR/'class_distribution.png')))""")
md("""The training set has 63,660 labelled images and the test set 29,443 unlabelled images. The two Eye classes contain
twice as many images as the other classes (a left and a right eye per animal), so the data are imbalanced 2:1 for
Eye vs non-Eye. Per species, the 4 classes are CatEye 21.8 %, others 10.9 % each.""")
code("display(Image(str(OUT_DIR/'samples_per_class.png')))")
code("display(Image(str(OUT_DIR/'size_by_class.png'))); display(Image(str(OUT_DIR/'blur.png'))); print(pd.read_csv(OUT_DIR/'size_stats.csv', header=[0,1], index_col=0))")
md("""**Data-quality findings.** Source crops range from 4 to 2219 px and are often non-square, with size strongly tied to
class (faces are large, mouths are wide). 15 exact duplicate files exist inside train (none with conflicting labels and
none shared with test); these are grouped by MD5 so they cannot straddle the train/validation split. Roughly 2 % of
images are under 24 px and a tail of blurry/dark crops exists. Mouth and Nose crops of the same species overlap visually
(unclear class boundary). Use: 90 % of labelled data to train, 10 % stratified hold-out to validate, test set only for
the Kaggle submission.""")

md("## 2. Preprocessing and pipeline (Task 2)")
code("""import inspect, pipeline
print(inspect.getsource(pipeline.make_split)); print(inspect.getsource(pipeline.augment))
print(open('src/cache_data.py').read()[:1500])""")
md("""All images (train and test) go through the identical deterministic path: decode -> RGB -> bicubic resize to 96x96
(cached as uint8) -> area-downscale to 64x64 -> /255 -> ImageNet mean/std normalisation. Labels are encoded to
integers 0-7. Augmentation (training only, applied on-the-fly to each batch): horizontal flip, rotation +/-15 deg,
scale 0.85-1.15, translation +/-10 %, brightness/contrast jitter. No vertical flips (faces are never upside down).
Because the split is performed before any statistics are fitted and the test data are only ever passed through the same
fixed transform, there is no leakage.""")

md("## 3. Model (Task 3)")
code("""import torch, timm
from pipeline import build_model
m = build_model('resnet18', pretrained=False); s = build_model('smallcnn')
print('ResNet-18 params: %.2fM | SmallCNN params: %.2fM' % (sum(p.numel() for p in m.parameters())/1e6, sum(p.numel() for p in s.parameters())/1e6))
print(m.fc)""")
md("""**Chosen model: ResNet-18 (timm `resnet18.a1_in1k` ImageNet weights)**, input 3x64x64, global-average-pool, dropout 0.2,
linear layer with 8 outputs, softmax inside the cross-entropy loss. Loss: cross-entropy with label smoothing 0.1.
Optimiser: AdamW (weight decay 0.05), OneCycle schedule (peak LR 2e-3, 20 % warm-up, cosine decay), batch size 128,
5 epochs. A from-scratch 4-block CNN (`SmallCNN`) is the baseline.""")

md("## 4. Training behaviour (Task 4)")
code("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for tag, ls in [('resnet18_64','-'), ('smallcnn_64','--')]:
    f = OUT_DIR/f'history_{tag}.json'
    if not f.exists(): continue
    h = pd.DataFrame(json.load(open(f)))
    ax[0].plot(h.epoch, h.train_loss, ls, label=f'{tag} train'); ax[0].plot(h.epoch, h.val_loss, ls, label=f'{tag} val')
    ax[1].plot(h.epoch, h.train_acc, ls, label=f'{tag} train'); ax[1].plot(h.epoch, h.val_acc, ls, label=f'{tag} val')
    print(tag); print(h[['epoch','train_loss','train_acc','val_loss','val_acc']].round(4).to_string(index=False))
ax[0].set_title('Loss'); ax[1].set_title('Accuracy'); ax[1].set_ylim(0.7, 1); [a.legend(fontsize=7) or a.set_xlabel('epoch') for a in ax]
plt.tight_layout(); plt.savefig(OUT_DIR/'learning_curves.png', dpi=150); plt.show()""")
md("""Training accuracy is *lower* than validation accuracy in early epochs because training batches are augmented and use
label smoothing, while validation images are clean. The gap closes by epoch 5 and neither model shows a rising
validation loss, so there is no sign of over-fitting; the from-scratch baseline is the one that is limited by
under-fitting/capacity at this budget.""")

md("## 5. Evaluation and Kaggle submission (Task 5)")
code("""for tag in ['resnet18_64','smallcnn_64']:
    f = OUT_DIR/f'val_metrics_{tag}.json'
    if f.exists(): print(tag, json.load(open(f)))
print(pd.read_csv(OUT_DIR/'classification_report_resnet18_64.csv', index_col=0))
display(Image(str(OUT_DIR/'confusion_resnet18_64.png')))
sub = pd.read_csv(OUT_DIR/'submission_resnet18_64.csv'); print(sub.shape); print(sub.head()); print(sub['class'].value_counts())""")
md("""**Kaggle leaderboard result:** `<ADD SCORE>` (private/public) - screenshot saved as `evidence/leaderboard.png`.
Compare with the validation accuracy above.""")

md("## 6. Error analysis (Task 6)")
code("display(Image(str(OUT_DIR/'errors_resnet18_64.png')))")
md("""Most errors are CatNose predicted as CatMouth (and vice versa): the Cat 'mouth' crop already contains the nose, so the
labels overlap. Eyes are almost never confused with anything else. See the report for discussion and improvement ideas.""")
nb['cells'] = C
nbf.write(nb, 'ENGE707_Facial_Feature_Classification.ipynb')
