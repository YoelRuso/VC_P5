"""Red neuronal: CNN pequeña (PyTorch) para emociones sobre FER2013 (6 clases, sin 'disgust').

Salida: modelos/cnn.pt y resultados/cnn_*.{json,png}
"""
import json
import math
import os
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from emociones_comun import EmotionCNN, a_tensor
from entrenar_svm import AQUI, cargar, nombres

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH = 128
LR = 2e-3
torch.manual_seed(0)
np.random.seed(0)


def aumentar(x):
    """Aumento de datos por lotes: giro horizontal, rotación ±12°, escala 0.9-1.1, traslación ±8 %."""
    n, d = x.size(0), x.device
    flip = (torch.rand(n, device=d) < 0.5).float() * 2 - 1
    ang = (torch.rand(n, device=d) * 2 - 1) * math.radians(12)
    esc = 0.9 + torch.rand(n, device=d) * 0.2
    tx = (torch.rand(n, device=d) * 2 - 1) * 0.08
    ty = (torch.rand(n, device=d) * 2 - 1) * 0.08
    cos, sin = torch.cos(ang) / esc, torch.sin(ang) / esc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def predecir(model, X, batch=512):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(X[i:i + batch].to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    Xtr, ytr = cargar("train")
    Xva, yva = cargar("validation")
    Xte, yte = cargar("test")
    Ttr, Tva, Tte = a_tensor(Xtr), a_tensor(Xva), a_tensor(Xte)
    Ttr = Ttr.to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)

    model = EmotionCNN(len(nombres)).to(DEVICE)
    print("Dispositivo:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "",
          flush=True)
    print("Parámetros:", sum(p.numel() for p in model.parameters()), flush=True)
    pesos = torch.tensor(len(ytr) / (len(nombres) * np.bincount(ytr)), dtype=torch.float32,
                         device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pasos = EPOCHS * math.ceil(len(Ttr) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=pasos)

    mejor, mejor_estado, hist = 0.0, None, []
    t_ini = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Ttr), device=DEVICE)
        perdida = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = aumentar(Ttr[idx]), ytr_t[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            perdida += loss.item() * len(idx)
        acc_va = accuracy_score(yva, predecir(model, Tva))
        hist.append({"epoca": ep, "loss": perdida / len(perm), "acc_val": acc_va})
        print(f"época {ep:2d}/{EPOCHS}  loss {perdida / len(perm):.3f}  val acc {acc_va:.3f}  "
              f"({time.time() - t_ini:.0f}s)", flush=True)
        if acc_va > mejor:
            mejor = acc_va
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
    t_train = time.time() - t_ini

    model.load_state_dict(mejor_estado)
    pred = predecir(model, Tte)
    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
    model = model.cpu().eval()  # la demo corre en CPU; medimos el tiempo de inferencia en CPU
    muestra = Tte[:200]
    t0 = time.time()
    with torch.no_grad():
        for i in range(200):
            model(muestra[i:i + 1])
    t_inf = (time.time() - t0) / 200 * 1000
    print(f"MEJOR VAL={mejor:.3f} | TEST acc={acc:.3f} f1_macro={f1:.3f} | {t_inf:.2f} ms/imagen")
    print(classification_report(yte, pred, target_names=nombres, digits=3))

    torch.save({"state_dict": model.state_dict(), "nombres": nombres},
               os.path.join(AQUI, "modelos", "cnn.pt"))
    with open(os.path.join(AQUI, "resultados", "cnn_metricas.json"), "w", encoding="utf-8") as f:
        json.dump({"acc_val": mejor, "acc_test": acc, "f1_macro_test": f1,
                   "ms_por_imagen": t_inf, "segundos_entrenamiento": t_train,
                   "epocas": EPOCHS, "historial": hist, "clases": nombres}, f, indent=2,
                  ensure_ascii=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred, normalize="true"),
                           display_labels=nombres).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                        colorbar=False)
    ax.set_title(f"CNN  (acc test {acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(AQUI, "resultados", "cnn_confusion.png"), dpi=130)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h["epoca"] for h in hist], [h["acc_val"] for h in hist], label="val acc")
    ax.plot([h["epoca"] for h in hist], [h["loss"] for h in hist], label="train loss")
    ax.set_xlabel("época")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(AQUI, "resultados", "cnn_curvas.png"), dpi=130)
    print("Listo.")
