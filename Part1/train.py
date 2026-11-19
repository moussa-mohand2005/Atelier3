import itertools
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

from models import ARCHS, SeqRegressor
from preprocessing import CLASSES, DATA, discretize

ROOT = Path(__file__).parent
RES = ROOT / "results"
SEED, MAX_LEN, MAX_VOCAB, BATCH = 42, 120, 20000, 64
MAX_EPOCHS, PATIENCE = 10, 2
GRID = {
    "emb_dim": [128],
    "hidden": [64, 128],
    "layers": [1, 2],
    "lr": [2e-3],
    "dropout": [0.3],
}
torch.set_num_threads(4)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vocab(token_lists):
    counts = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in counts.most_common(MAX_VOCAB) if c >= 2]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(token_lists, vocab):
    X = np.zeros((len(token_lists), MAX_LEN), dtype=np.int64)
    L = np.zeros(len(token_lists), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = [vocab.get(t, 1) for t in toks][:MAX_LEN] or [1]
        X[i, : len(ids)] = ids
        L[i] = len(ids)
    return torch.tensor(X), torch.tensor(L)


def batches(X, L, y, shuffle):
    idx = torch.randperm(len(y)) if shuffle else torch.arange(len(y))
    for s in range(0, len(y), BATCH):
        b = idx[s : s + BATCH]
        yield X[b], L[b], y[b]


@torch.no_grad()
def predict(model, X, L):
    model.eval()
    return torch.cat([model(xb, lb) for xb, lb, _ in batches(X, L, torch.zeros(len(L)), False)]).numpy()


def train_one(arch, cfg, vocab_size, train, val):
    seed_all()
    model = SeqRegressor(arch, vocab_size, cfg["emb_dim"], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tot = 0.0
        for xb, lb, yb in batches(*train, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
        train_mse = tot / len(train[2])
        val_mse = mean_squared_error(val[2].numpy(), predict(model, val[0], val[1]))
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best - 1e-4:
            best, bad = val_mse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, history


def label_bleu(y_true_cls, y_pred_cls):
    ref = [CLASSES[c] for c in y_true_cls]
    hyp = [CLASSES[c] for c in y_pred_cls]
    refs = [[ref[i : i + 10]] for i in range(0, len(ref), 10)]
    hyps = [hyp[i : i + 10] for i in range(0, len(hyp), 10)]
    sm = SmoothingFunction().method1
    return (corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=sm),
            corpus_bleu(refs, hyps, smoothing_function=sm))


def evaluate(y, p):
    yc, pc = [discretize(v) for v in y], [discretize(v) for v in p]
    b1, b4 = label_bleu(yc, pc)
    return {
        "MSE": mean_squared_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
        "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p), "Pearson": pearsonr(y, p)[0],
        "Accuracy(3 classes)": accuracy_score(yc, pc), "F1-macro": f1_score(yc, pc, average="macro"),
        "BLEU-1": b1, "BLEU-4": b4,
    }, (yc, pc)


def main():
    RES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "dataset.csv")
    tokens = json.loads((DATA / "tokens.json").read_text("utf-8"))
    y_all = df.Score.to_numpy(dtype=np.float32)
    strat = [discretize(s) for s in y_all]
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=strat)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=[strat[i] for i in tmp])
    vocab = build_vocab([tokens[i] for i in tr])
    X, L = encode(tokens, vocab)
    y = torch.tensor(y_all)
    split = lambda ids: (X[ids], L[ids], y[ids])
    train, val, test = split(tr), split(va), split(te)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} | vocab {len(vocab) + 2}")

    tuning, metrics, histories, preds = [], [], {}, {"Text": df.Text.iloc[te].values, "true": y_all[te]}
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    for arch in ARCHS:
        best = None
        for cfg in configs:
            t0 = time.time()
            model, val_mse, hist = train_one(arch, cfg, len(vocab) + 2, train, val)
            dt = time.time() - t0
            tuning.append({"arch": arch, **cfg, "val_RMSE": val_mse ** 0.5, "epochs": len(hist), "seconds": round(dt, 1)})
            print(f"{arch:6s} {cfg} val_RMSE={val_mse ** 0.5:.3f} ({len(hist)} ep, {dt:.0f}s)", flush=True)
            if best is None or val_mse < best[1]:
                best = (model, val_mse, hist, cfg)
        model, _, hist, cfg = best
        torch.save({"state_dict": model.state_dict(), "config": cfg, "vocab": vocab}, RES / f"best_{arch}.pt")
        p = predict(model, test[0], test[1])
        m, (yc, pc) = evaluate(y_all[te], p)
        n_params = sum(t.numel() for t in model.parameters())
        metrics.append({"model": arch, **{k: round(float(v), 4) for k, v in m.items()},
                        "params": n_params, "best_config": json.dumps(cfg)})
        histories[arch], preds[arch] = hist, np.round(p, 2)
        preds[arch + "_cm"] = confusion_matrix(yc, pc, labels=[0, 1, 2])

    pd.DataFrame(tuning).to_csv(RES / "tuning.csv", index=False)
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(RES / "metrics.csv", index=False)
    pd.DataFrame({k: v for k, v in preds.items() if not k.endswith("_cm")}).to_csv(
        RES / "predictions.csv", index=False, encoding="utf-8")
    print(mdf.drop(columns="best_config").to_string(index=False))
    plots(histories, preds, y_all[te])


def plots(histories, preds, y_true):
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6), sharey=True)
    for ax, (arch, h) in zip(axes, histories.items()):
        ax.plot([e["epoch"] for e in h], [e["train_mse"] for e in h], label="train")
        ax.plot([e["epoch"] for e in h], [e["val_mse"] for e in h], label="val")
        ax.set_title(arch); ax.set_xlabel("epoch")
    axes[0].set_ylabel("MSE"); axes[0].legend()
    fig.tight_layout(); fig.savefig(RES / "loss_curves.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, arch in zip(axes, histories):
        ax.scatter(y_true, preds[arch], s=6, alpha=0.4)
        ax.plot([0, 10], [0, 10], "k--", lw=1)
        ax.set_title(arch); ax.set_xlabel("true score")
    axes[0].set_ylabel("predicted score")
    fig.tight_layout(); fig.savefig(RES / "pred_vs_true.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, arch in zip(axes, histories):
        cm = preds[arch + "_cm"]
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
        ax.set_title(arch); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(RES / "confusion_matrices.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
