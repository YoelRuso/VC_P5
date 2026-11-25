"""Convierte FER2013 (copia de Hugging Face) a un .npz ligero y muestra la distribución de clases."""
import os

import numpy as np
from datasets import load_dataset

AQUI = os.path.dirname(os.path.abspath(__file__))
ds = load_dataset("abhilash88/fer2013-enhanced", cache_dir=os.path.join(AQUI, "datos", "hf_cache"))
nombres = ds["train"].features["emotion"].names
out = {}
for split in ("train", "validation", "test"):
    X = np.array(ds[split]["image"], dtype=np.uint8)
    y = np.array(ds[split]["emotion"], dtype=np.int64)
    out[f"X_{split}"], out[f"y_{split}"] = X, y
    cuentas = {nombres[i]: int((y == i).sum()) for i in range(len(nombres))}
    print(split, X.shape, cuentas)
np.savez_compressed(os.path.join(AQUI, "datos", "fer2013.npz"), nombres=np.array(nombres), **out)
print("Guardado datos/fer2013.npz")
