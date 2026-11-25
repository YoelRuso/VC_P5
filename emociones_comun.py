"""Código compartido entre el entrenamiento y la demo en directo (modelo de emociones)."""
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import hog


class EmotionCNN(nn.Module):
    """3 bloques conv (32-64-128) + clasificador denso. Entrada 1x48x48 normalizada a [-1, 1]."""

    def __init__(self, n_clases=6):
        super().__init__()

        def bloque(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25))

        self.features = nn.Sequential(bloque(1, 32), bloque(32, 64), bloque(64, 128))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(),
                                        nn.Dropout(0.5), nn.Linear(256, n_clases))

    def forward(self, x):
        return self.classifier(self.features(x))


def a_tensor(X):
    """uint8 (N,48,48) -> float (N,1,48,48) en [-1, 1]."""
    return (torch.from_numpy(np.asarray(X)).float().unsqueeze(1) - 127.5) / 127.5


def hog_features(imgs):
    """Características HOG (1764 por imagen de 48x48) usadas por el SVM."""
    return np.array([hog(im, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2),
                         block_norm="L2-Hys") for im in imgs], dtype=np.float32)
