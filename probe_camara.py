"""Diagnóstico de cámara: muestra qué índices/backends entregan imagen en ESTE terminal."""
import sys
import time

import cv2

print("Python:", sys.executable)
print("OpenCV:", cv2.__version__)
found = False
for name, be in (("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)):
    for i in range(4):
        cap = cv2.VideoCapture(i, be)
        if not cap.isOpened():
            cap.release()
            continue
        got = None
        for _ in range(20):
            ok, frame = cap.read()
            if ok:
                got = frame.shape
                break
            time.sleep(0.1)
        print(f"{name} indice {i}: abierta,", f"IMAGEN {got}" if got else "SIN imagen")
        found = found or got is not None
        cap.release()
print("RESULTADO:", "hay cámara con imagen" if found else "ninguna cámara da imagen")
