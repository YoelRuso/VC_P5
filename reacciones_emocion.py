"""
Prototipo 1: "Espejo de emociones" (VC - Práctica 5)

MediaPipe localiza la cara y sus landmarks; se recorta la cara, se pasa a 48x48 en gris y un modelo
ENTRENADO POR NOSOTROS (CNN o HOG+SVM, sobre FER2013) clasifica la emoción. El resultado dispara
efectos anclados a los landmarks:
    alegría   -> confeti
    tristeza  -> lágrimas que salen de los ojos
    enfado    -> tinte rojo + vapor sobre la cabeza
    sorpresa  -> ojos de dibujo animado que se agrandan
    miedo     -> viñeta oscura + gotas de sudor en la frente
    neutral   -> sin efecto

Teclas:  M = cambiar de modelo (CNN <-> SVM)   Q / ESC = salir
Variables de entorno opcionales: CAMERA_INDEX, CAMERA_ROTATE (cw, ccw, 180, none).
"""
import math
import os
import random
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from emociones_comun import EmotionCNN, a_tensor, hog_features
from tetris_cara import CAMERA_ROTATE, MODEL_PATH, MODEL_URL, open_camera  # misma cámara que el Tetris

AQUI = os.path.dirname(os.path.abspath(__file__))
ESCALA = 1.5            # ampliación de la ventana
SUAVIZADO = 0.35        # peso de la predicción nueva en la media móvil de probabilidades
NOMBRES_ES = {"angry": "ENFADO", "fear": "MIEDO", "happy": "ALEGRIA", "sad": "TRISTEZA",
              "surprise": "SORPRESA", "neutral": "NEUTRAL"}
COLORES = {"angry": (60, 60, 255), "fear": (200, 80, 160), "happy": (0, 220, 255),
           "sad": (255, 150, 60), "surprise": (255, 255, 255), "neutral": (170, 170, 170)}


# --------------------------------------------------------------------------- #
# Modelos entrenados
# --------------------------------------------------------------------------- #
class Clasificador:
    def __init__(self):
        # El checkpoint lo generamos nosotros (entrenar_cnn.py), por eso weights_only=False.
        ck = torch.load(os.path.join(AQUI, "modelos", "cnn.pt"), map_location="cpu",
                        weights_only=False)
        self.nombres = [str(n) for n in ck["nombres"]]
        self.cnn = EmotionCNN(len(self.nombres))
        self.cnn.load_state_dict(ck["state_dict"])
        self.cnn.eval()
        ruta_svm = os.path.join(AQUI, "modelos", "svm_hog.joblib")
        self.svm = joblib.load(ruta_svm)["clf"] if os.path.exists(ruta_svm) else None
        self.usar_cnn = True

    @property
    def nombre_modelo(self):
        return "CNN" if self.usar_cnn or self.svm is None else "HOG+SVM"

    def alternar(self):
        if self.svm is not None:
            self.usar_cnn = not self.usar_cnn

    def probabilidades(self, gris48):
        if self.usar_cnn or self.svm is None:
            with torch.no_grad():
                return torch.softmax(self.cnn(a_tensor(gris48[None])), 1)[0].numpy()
        s = self.svm.decision_function(hog_features(gris48[None]))[0]
        e = np.exp((s - s.max()) * 2.0)
        return e / e.sum()


# --------------------------------------------------------------------------- #
# Seguimiento de la cara (landmarks)
# --------------------------------------------------------------------------- #
class Landmarks:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.t0, self.last = time.time(), -1

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ts = max(int((time.time() - self.t0) * 1000), self.last + 1)
        self.last = ts
        res = self.lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)


def recortar_cara(frame_bgr, pts):
    """Recorte cuadrado de la cara (de frente a barbilla) a 48x48 en gris, como FER2013."""
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lado = max(x1 - x0, y1 - y0) * 1.05
    h, w = frame_bgr.shape[:2]
    xa, xb = int(max(cx - lado / 2, 0)), int(min(cx + lado / 2, w))
    ya, yb = int(max(cy - lado / 2, 0)), int(min(cy + lado / 2, h))
    if xb - xa < 20 or yb - ya < 20:
        return None
    gris = cv2.cvtColor(frame_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gris, (48, 48), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Efectos anclados a landmarks
# --------------------------------------------------------------------------- #
class Particulas:
    def __init__(self):
        self.p = []

    def emitir(self, x, y, vx, vy, vida, color, r):
        self.p.append([x, y, vx, vy, vida, color, r])

    def actualizar_y_dibujar(self, frame, g=0.0):
        vivas = []
        for q in self.p:
            q[0] += q[2]
            q[1] += q[3]
            q[3] += g
            q[4] -= 1
            if q[4] > 0:
                cv2.circle(frame, (int(q[0]), int(q[1])), int(q[6]), q[5], -1, cv2.LINE_AA)
                vivas.append(q)
        self.p = vivas[-400:]


def tinte(frame, color, alfa):
    cv2.addWeighted(np.full_like(frame, color), alfa, frame, 1 - alfa, 0, frame)


def vineta(frame, fuerza=0.75):
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    m = np.clip(1 - fuerza * np.clip(d - 0.4, 0, 1), 0, 1)[..., None]
    frame[:] = (frame * m).astype(np.uint8)


def aplicar_efecto(frame, emocion, pts, parts, t):
    h, w = frame.shape[:2]
    ojo_i, ojo_d = pts[468], pts[473]               # centros de los iris
    frente = pts[10]
    ancho = float(np.linalg.norm(pts[454] - pts[234]))  # ancho de la cara
    if emocion == "happy":
        for _ in range(4):
            parts.emitir(random.uniform(0, w), -5, random.uniform(-1, 1), random.uniform(3, 7), 140,
                         random.choice([(0, 220, 255), (255, 120, 255), (120, 255, 120),
                                        (255, 200, 80)]), random.randint(3, 6))
        parts.actualizar_y_dibujar(frame, g=0.1)
    elif emocion == "sad":
        tinte(frame, (150, 90, 40), 0.18)
        for o in (ojo_i, ojo_d):
            if random.random() < 0.5:
                parts.emitir(o[0] + random.uniform(-4, 4), o[1] + ancho * 0.07, 0,
                             random.uniform(1.0, 2.0), 50, (255, 190, 90), max(3, int(ancho * 0.02)))
        parts.actualizar_y_dibujar(frame, g=0.25)
    elif emocion == "angry":
        tinte(frame, (0, 0, 220), 0.25)
        for _ in range(2):
            parts.emitir(frente[0] + random.uniform(-ancho * 0.3, ancho * 0.3), frente[1] - ancho * 0.05,
                         random.uniform(-0.6, 0.6), random.uniform(-3, -1.5), 45, (200, 200, 200),
                         max(5, int(ancho * 0.05)))
        parts.actualizar_y_dibujar(frame)
    elif emocion == "surprise":
        r = int(ancho * (0.17 + 0.03 * math.sin(t * 12)))
        for o in (ojo_i, ojo_d):
            c = (int(o[0]), int(o[1]))
            cv2.circle(frame, c, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, c, r, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(frame, c, max(3, r // 3), (0, 0, 0), -1, cv2.LINE_AA)
    elif emocion == "fear":
        vineta(frame, 0.8)
        tinte(frame, (140, 40, 90), 0.12)
        if random.random() < 0.4:
            parts.emitir(frente[0] + random.uniform(-ancho * 0.25, ancho * 0.25), frente[1] + ancho * 0.05,
                         0, random.uniform(1, 2.5), 45, (255, 230, 120), max(3, int(ancho * 0.025)))
        parts.actualizar_y_dibujar(frame, g=0.15)
    else:
        parts.p.clear()


def dibujar_hud(frame, nombres, probs, modelo, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (190, 30 + 24 * len(nombres)), (20, 20, 20), -1)
    cv2.putText(frame, f"{modelo}  {fps:.0f} fps", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    for i, n in enumerate(nombres):
        y = 40 + 24 * i
        cv2.rectangle(frame, (8, y), (8 + int(120 * probs[i]), y + 12), COLORES[n], -1)
        cv2.putText(frame, f"{NOMBRES_ES[n][:6]} {probs[i] * 100:3.0f}%", (8, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    clasif = Clasificador()
    nombres = clasif.nombres
    cap = open_camera()
    if cap is None:
        raise SystemExit("No se encontró cámara. Revisa DroidCam (Start) y cierra otras apps.")
    tracker = Landmarks()
    probs = np.ones(len(nombres)) / len(nombres)
    parts = Particulas()
    rot = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
           "180": cv2.ROTATE_180}.get(CAMERA_ROTATE)
    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        if rot is not None:
            frame = cv2.rotate(frame, rot)
        frame = cv2.flip(frame, 1)
        pts = tracker(frame)
        emocion = "neutral"
        if pts is not None:
            cara = recortar_cara(frame, pts)
            if cara is not None:
                probs = (1 - SUAVIZADO) * probs + SUAVIZADO * clasif.probabilidades(cara)
            emocion = nombres[int(np.argmax(probs))]
            aplicar_efecto(frame, emocion, pts, parts, time.time())
            x0, y0 = pts.min(0).astype(int)
            x1, y1 = pts.max(0).astype(int)
            cv2.rectangle(frame, (x0, y0), (x1, y1), COLORES[emocion], 2)
            cv2.putText(frame, NOMBRES_ES[emocion], (x0, max(y0 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, COLORES[emocion], 2, cv2.LINE_AA)
        else:
            parts.p.clear()
        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - t_prev, 1e-6)
        t_prev = ahora
        dibujar_hud(frame, nombres, probs, clasif.nombre_modelo, fps)
        frame = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA)
        cv2.imshow("Espejo de emociones", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("m"):
            clasif.alternar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
