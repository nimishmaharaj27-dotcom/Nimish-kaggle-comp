import argparse
from pipeline import train

p = argparse.ArgumentParser()
p.add_argument("--model", default="resnet18")
p.add_argument("--size", type=int, default=64)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--no-pretrained", action="store_true")
p.add_argument("--tag", default=None)
p.add_argument("--max-train", type=int, default=None)
a = p.parse_args()
train(a.model, a.size, a.epochs, a.bs, a.lr, pretrained=not a.no_pretrained, tag=a.tag, max_train=a.max_train)
