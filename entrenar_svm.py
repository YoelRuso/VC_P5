"""Modelo clásico: HOG + SVM (RBF) para emociones sobre FER2013 (6 clases, sin 'disgust').

Salida: modelos/svm_hog.joblib y resultados/svm_*.{json,png}
"""
import json
import os
import time

import joblib
import matplotlib
import numpy as np
from emociones_comun import hog_features
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(AQUI, "modelos"), exist_ok=True)
os.makedirs(os.path.join(AQUI, "resultados"), exist_ok=True)

d = np.load(os.path.join(AQUI, "datos", "fer2013.npz"))
nombres_orig = [str(n) for n in d["nombres"]]
DESCARTAR = nombres_orig.index("disgust")
nombres = [n for n in nombres_orig if n != "disgust"]


def cargar(split):
    X, y = d[f"X_{split}"], d[f"y_{split}"]
    m = y != DESCARTAR
    X, y = X[m], y[m]
    y = np.where(y > DESCARTAR, y - 1, y)  # reindexa las clases tras quitar 'disgust'
    return X, y


caracteristicas = hog_features


if __name__ == "__main__":
    Xtr, ytr = cargar("train")
    Xva, yva = cargar("validation")
    Xte, yte = cargar("test")
    print("Clases:", nombres, "| train:", len(ytr), "val:", len(yva), "test:", len(yte))

    t0 = time.time()
    Ftr, Fva, Fte = caracteristicas(Xtr), caracteristicas(Xva), caracteristicas(Xte)
    print(f"HOG: {Ftr.shape[1]} características por imagen ({time.time() - t0:.0f}s)")

    t0 = time.time()
    clf = SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced")
    clf.fit(Ftr, ytr)
    t_train = time.time() - t0
    print(f"Entrenado en {t_train:.0f}s")

    acc_val = accuracy_score(yva, clf.predict(Fva))
    t0 = time.time()
    pred = clf.predict(Fte)
    t_inf = (time.time() - t0) / len(Fte) * 1000
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    print(f"VAL acc={acc_val:.3f} | TEST acc={acc:.3f} f1_macro={f1:.3f} | {t_inf:.2f} ms/imagen (solo SVM)")
    print(classification_report(yte, pred, target_names=nombres, digits=3))

    joblib.dump({"clf": clf, "nombres": nombres}, os.path.join(AQUI, "modelos", "svm_hog.joblib"))
    with open(os.path.join(AQUI, "resultados", "svm_metricas.json"), "w", encoding="utf-8") as f:
        json.dump({"acc_val": acc_val, "acc_test": acc, "f1_macro_test": f1,
                   "ms_por_imagen_clasificador": t_inf, "segundos_entrenamiento": t_train,
                   "clases": nombres}, f, indent=2, ensure_ascii=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred, normalize="true"),
                           display_labels=nombres).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                        colorbar=False)
    ax.set_title(f"HOG + SVM  (acc test {acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(AQUI, "resultados", "svm_confusion.png"), dpi=130)
    print("Listo.")
