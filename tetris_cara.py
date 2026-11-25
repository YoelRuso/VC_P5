"""
Tetris controlado con la cara (VC - Práctica 5, prototipo libre)

Controles faciales (MediaPipe Face Landmarker):
  - Inclinar la cabeza (roll)  -> mover la pieza a izquierda / derecha
  - Parpadeo deliberado        -> rotar la pieza (cerrar ambos ojos ~0,3 s)
  - Levantar las cejas         -> caída rápida (hard drop)

Teclado (para depurar sin cámara): flechas izq/der, flecha arriba (rotar),
espacio (caída rápida), C (recalibrar), R (reiniciar), ESC (salir).

Dependencias:
    pip install mediapipe opencv-python pygame numpy
El modelo face_landmarker.task se descarga solo la primera vez que se ejecuta.
"""
import math
import os
import random
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------- #
# Configuración (ajusta estos valores si el control no te va fino)
# --------------------------------------------------------------------------- #
CAMERA_INDEX = 0
ROLL_ON = 8.0          # grados de inclinación para dar el primer paso
ROLL_OFF = 5.0         # por debajo de esto se "suelta" (histéresis: evita pasos por temblor)
ROLL_MAX = 22.0        # inclinación a la que se alcanza la velocidad máxima
REPEAT_SLOW = 0.50     # s entre pasos justo al pasar ROLL_ON (control fino, 1 columna cada vez)
REPEAT_FAST = 0.10     # s entre pasos con inclinación máxima (desplazamiento rápido)
ROLL_SMOOTH = 0.45     # 0..1: peso de la medida nueva en el filtro (menos = más suave)
CAMERA_ROTATE = os.environ.get("CAMERA_ROTATE", "cw")  # "cw", "ccw", "180" o "none" (DroidCam en vertical)
ROLL_SIGN = -1        # 1 o -1: invierte el sentido del movimiento lateral
BLINK_HOLD = 0.12      # segundos con los ojos cerrados para contar como "rotar"
BLINK_COOLDOWN = 0.40  # segundos mínimos entre dos rotaciones
BLINK_THR_FRAC = 0.35  # fracción del recorrido ojo abierto -> cerrado que cuenta como parpadeo
BROW_COOLDOWN = 0.8    # segundos mínimos entre dos caídas rápidas
STEP_TIME = 3.0        # duración de cada paso de calibración
SAMPLE_FROM = 1.0      # se promedia solo lo medido después de este segundo del paso

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# --------------------------------------------------------------------------- #
# Seguimiento facial
# --------------------------------------------------------------------------- #
class FaceTracker:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print("Descargando face_landmarker.task ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.t0 = time.time()
        self.last_ts = -1

    def process(self, frame_bgr):
        """Devuelve dict con roll (grados), blink, brow (0..1) y puntos para dibujar, o None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.time() - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        res = self.landmarker.detect_for_video(img, ts)
        if not res.face_landmarks or not res.face_blendshapes:
            return None
        lms = res.face_landmarks[0]
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}

        # Roll: ángulo de la línea entre las comisuras externas de los ojos (33 y 263).
        # Se ordenan por x para que no dependa de qué lado es cada ojo.
        p1 = (lms[33].x * w, lms[33].y * h)
        p2 = (lms[263].x * w, lms[263].y * h)
        left, right = (p1, p2) if p1[0] < p2[0] else (p2, p1)
        roll = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))

        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        brow = (bs.get("browInnerUp", 0) + bs.get("browOuterUpLeft", 0)
                + bs.get("browOuterUpRight", 0)) / 3
        return {"roll": roll, "blink": blink, "brow": brow,
                "eyes": (int(left[0]), int(left[1]), int(right[0]), int(right[1]))}


# --------------------------------------------------------------------------- #
# Calibración y traducción de gestos a eventos de juego
# --------------------------------------------------------------------------- #
class FaceController:
    STEPS = ["Cara neutra, mira a la cámara", "Cierra los ojos", "Levanta las cejas"]

    def __init__(self):
        self.start_calibration()
        self.reset_events()

    def start_calibration(self):
        self.calibrating = True
        self.step = 0
        self.step_t0 = None
        self.samples = []
        self.neutral = None
        self.blink_closed = None
        self.brow_raised = None
        self.roll0 = 0.0
        self.blink_thr = 0.5
        self.brow_thr = 0.5

    def reset_events(self):
        self.last_dir = 0
        self.last_move = 0.0
        self.dr_s = None
        self.closed_since = None
        self.blink_fired = False
        self.brow_active = False
        self.last_drop = 0.0
        self.last_rotate = 0.0

    def calibrate(self, feat, now):
        """Avanza la calibración. Devuelve (texto, progreso 0..1)."""
        if self.step_t0 is None:
            self.step_t0 = now
        t = now - self.step_t0
        if feat is not None and t >= SAMPLE_FROM:
            self.samples.append((feat["roll"], feat["blink"], feat["brow"]))
        if t >= STEP_TIME:
            arr = np.array(self.samples) if self.samples else np.zeros((1, 3))
            mean = arr.mean(axis=0)
            if self.step == 0:
                self.neutral = mean
                self.roll0 = float(mean[0])
            elif self.step == 1:
                self.blink_closed = mean[1]
            else:
                self.brow_raised = mean[2]
            self.step += 1
            self.step_t0 = now
            self.samples = []
            if self.step >= len(self.STEPS):
                self._finish_calibration()
                return "Calibración lista", 1.0
            t = 0.0
        return self.STEPS[self.step], min(t / STEP_TIME, 1.0)

    def _finish_calibration(self):
        n_blink, n_brow = self.neutral[1], self.neutral[2]
        # Umbral = punto medio entre neutro y gesto, con un hueco mínimo por seguridad.
        self.blink_thr = n_blink + max((self.blink_closed - n_blink) * BLINK_THR_FRAC, 0.12)
        self.brow_thr = n_brow + max((self.brow_raised - n_brow) * 0.5, 0.10)
        self.calibrating = False
        self.reset_events()

    def tilt(self, feat):
        """Inclinación en grados respecto a la postura neutra (positivo = mover a la derecha)."""
        return (feat["roll"] - self.roll0) * ROLL_SIGN

    def events(self, feat, now):
        """Devuelve (movimiento -1/0/1, rotar, caer)."""
        move, rotate, drop = 0, False, False
        if feat is None:
            self.reset_events()
            return move, rotate, drop

        # Movimiento lateral por inclinación de cabeza:
        #  - un golpe rápido de cabeza = exactamente 1 columna
        #  - mantener una inclinación suave = 1 columna cada ~0,5 s
        #  - inclinar más = más rápido (hasta 1 columna cada 0,1 s)
        raw = self.tilt(feat)
        self.dr_s = raw if self.dr_s is None else self.dr_s + ROLL_SMOOTH * (raw - self.dr_s)
        dr = self.dr_s
        mag, s = abs(dr), (1 if dr > 0 else -1)
        if mag > ROLL_ON:
            direction = s
        elif mag > ROLL_OFF and self.last_dir == s:
            direction = s          # histéresis: sigue activo hasta bajar de ROLL_OFF
        else:
            direction = 0
        if direction == 0:
            self.last_dir = 0
        elif direction != self.last_dir:
            move = direction       # primer paso inmediato al cruzar el umbral
            self.last_dir = direction
            self.last_move = now
        else:
            frac = min(max((mag - ROLL_ON) / (ROLL_MAX - ROLL_ON), 0.0), 1.0)
            interval = REPEAT_SLOW + (REPEAT_FAST - REPEAT_SLOW) * frac
            if now - self.last_move >= interval:
                move = direction
                self.last_move = now

        # Rotación por parpadeo mantenido (así no se confunde con un parpadeo natural)
        if feat["blink"] > self.blink_thr:
            if self.closed_since is None:
                self.closed_since = now
            elif (not self.blink_fired and now - self.closed_since >= BLINK_HOLD
                  and now - self.last_rotate >= BLINK_COOLDOWN):
                rotate = True
                self.blink_fired = True
                self.last_rotate = now
        else:
            self.closed_since = None
            self.blink_fired = False

        # Caída rápida al levantar las cejas (flanco de subida + enfriamiento)
        raised = feat["brow"] > self.brow_thr
        if raised and not self.brow_active and now - self.last_drop >= BROW_COOLDOWN:
            drop = True
            self.last_drop = now
        self.brow_active = raised
        return move, rotate, drop


# --------------------------------------------------------------------------- #
# Lógica del Tetris
# --------------------------------------------------------------------------- #
SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
}
COLORS = {
    "I": (0, 240, 240), "O": (240, 240, 0), "T": (170, 0, 240), "S": (0, 240, 0),
    "Z": (240, 0, 0), "J": (0, 80, 240), "L": (240, 160, 0),
}


def rotate_matrix(m):
    return [list(r) for r in zip(*m[::-1])]


class Tetris:
    W, H = 10, 20

    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = [[None] * self.W for _ in range(self.H)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.over = False
        self.bag = []
        self.next_kind = self._draw()
        self._spawn()
        self.last_fall = time.time()

    def _draw(self):
        if not self.bag:
            self.bag = list(SHAPES)
            random.shuffle(self.bag)
        return self.bag.pop()

    def _spawn(self):
        self.kind = self.next_kind
        self.next_kind = self._draw()
        self.shape = [row[:] for row in SHAPES[self.kind]]
        self.x = self.W // 2 - len(self.shape[0]) // 2
        self.y = 0
        if self._collides(self.shape, self.x, self.y):
            self.over = True

    def _collides(self, shape, x, y):
        for r, row in enumerate(shape):
            for c, v in enumerate(row):
                if not v:
                    continue
                gx, gy = x + c, y + r
                if gx < 0 or gx >= self.W or gy >= self.H:
                    return True
                if gy >= 0 and self.grid[gy][gx]:
                    return True
        return False

    def move(self, dx):
        if not self.over and not self._collides(self.shape, self.x + dx, self.y):
            self.x += dx

    def rotate(self):
        if self.over:
            return
        new = rotate_matrix(self.shape)
        for kick in (0, -1, 1, -2, 2):  # pequeños desplazamientos si choca con la pared
            if not self._collides(new, self.x + kick, self.y):
                self.shape, self.x = new, self.x + kick
                return

    def step(self):
        if not self._collides(self.shape, self.x, self.y + 1):
            self.y += 1
        else:
            self._lock()

    def hard_drop(self):
        if self.over:
            return
        while not self._collides(self.shape, self.x, self.y + 1):
            self.y += 1
            self.score += 2
        self._lock()

    def _lock(self):
        for r, row in enumerate(self.shape):
            for c, v in enumerate(row):
                if v and 0 <= self.y + r < self.H:
                    self.grid[self.y + r][self.x + c] = self.kind
        full = [i for i, row in enumerate(self.grid) if all(row)]
        for i in full:
            del self.grid[i]
            self.grid.insert(0, [None] * self.W)
        if full:
            self.lines += len(full)
            self.score += [0, 100, 300, 500, 800][len(full)] * self.level
            self.level = self.lines // 10 + 1
        self._spawn()
        self.last_fall = time.time()

    def update(self, now):
        if self.over:
            return
        interval = max(0.1, 0.8 - 0.07 * (self.level - 1))
        if now - self.last_fall >= interval:
            self.step()
            self.last_fall = now

    def ghost_y(self):
        y = self.y
        while not self._collides(self.shape, self.x, y + 1):
            y += 1
        return y


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
CELL = 30
BOARD_X, BOARD_Y = 30, 40
CAM_W, CAM_H = 360, 270
CAM_X, CAM_Y = 380, 40
WIN_W, WIN_H = 780, 680

BG = (18, 18, 28)
GRID_LINE = (40, 40, 58)
TEXT = (230, 230, 240)
ACCENT = (0, 220, 160)


def draw_block(surf, x, y, color, alpha=None):
    rect = pygame.Rect(x, y, CELL - 1, CELL - 1)
    if alpha is None:
        pygame.draw.rect(surf, color, rect)
        pygame.draw.rect(surf, tuple(min(255, v + 60) for v in color), rect, 2)
    else:
        pygame.draw.rect(surf, tuple(int(v * alpha) for v in color), rect, 2)


def draw_bar(surf, font, label, value, thr, x, y, w=300, color=ACCENT):
    surf.blit(font.render(label, True, TEXT), (x, y))
    bar = pygame.Rect(x, y + 22, w, 12)
    pygame.draw.rect(surf, GRID_LINE, bar)
    pygame.draw.rect(surf, color, (x, y + 22, int(w * max(0.0, min(1.0, value))), 12))
    if thr is not None:
        tx = x + int(w * max(0.0, min(1.0, thr)))
        pygame.draw.line(surf, (255, 80, 80), (tx, y + 18), (tx, y + 40), 2)


def draw_game(screen, fonts, game):
    for r in range(game.H):
        for c in range(game.W):
            rect = pygame.Rect(BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, CELL)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)
            k = game.grid[r][c]
            if k:
                draw_block(screen, rect.x, rect.y, COLORS[k])
    if not game.over:
        gy = game.ghost_y()
        for r, row in enumerate(game.shape):
            for c, v in enumerate(row):
                if v:
                    draw_block(screen, BOARD_X + (game.x + c) * CELL,
                               BOARD_Y + (gy + r) * CELL, COLORS[game.kind], alpha=0.6)
        for r, row in enumerate(game.shape):
            for c, v in enumerate(row):
                if v and game.y + r >= 0:
                    draw_block(screen, BOARD_X + (game.x + c) * CELL,
                               BOARD_Y + (game.y + r) * CELL, COLORS[game.kind])
    pygame.draw.rect(screen, TEXT, (BOARD_X - 2, BOARD_Y - 2, game.W * CELL + 4,
                                    game.H * CELL + 4), 2)


def _try_open_camera():
    env = os.environ.get("CAMERA_INDEX")
    indices = [int(env)] if env else list(range(0, 6))
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        for i in indices:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                for _ in range(15):  # algunas cámaras virtuales tardan en arrancar
                    ok, _frame = cap.read()
                    if ok:
                        print(f"Cámara encontrada: índice {i}, backend {backend}")
                        return cap
                    time.sleep(0.1)
            cap.release()
    return None


def open_camera(timeout=12.0):
    """Reintenta durante `timeout` segundos (DroidCam puede tardar en reconectar con el móvil)."""
    t0 = time.time()
    while True:
        cap = _try_open_camera()
        if cap is not None or time.time() - t0 > timeout:
            return cap
        print("Esperando a la cámara... (¿DroidCam conectado y con Start pulsado?)")
        time.sleep(1.0)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Tetris facial")
    fonts = {"s": pygame.font.SysFont("consolas", 16),
             "m": pygame.font.SysFont("consolas", 22, bold=True),
             "l": pygame.font.SysFont("consolas", 40, bold=True)}
    clock = pygame.time.Clock()

    cap = open_camera()
    if cap is None:
        raise SystemExit("No se encontró ninguna cámara que entregue imagen. "
                         "Con DroidCam: abre el cliente de PC, pulsa Start y cierra otras apps "
                         "que usen la cámara. Puedes forzar el índice con: "
                         "$env:CAMERA_INDEX=1; python tetris_cara.py")
    tracker = FaceTracker()
    ctrl = FaceController()
    game = Tetris()
    cam_surface = None
    last_ok = time.time()

    running = True
    while running:
        now = time.time()
        key_events = []
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    game.reset()
                elif e.key == pygame.K_c:
                    ctrl.start_calibration()
                else:
                    key_events.append(e.key)

        feat = None
        ok, frame = cap.read()
        if ok:
            last_ok = now
        elif now - last_ok > 3.0:  # la cámara se ha cortado: intenta reabrirla
            cap.release()
            new_cap = _try_open_camera()
            if new_cap is not None:
                cap = new_cap
            last_ok = now
        if ok:
            rot = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
                   "180": cv2.ROTATE_180}.get(CAMERA_ROTATE)
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            frame = cv2.flip(frame, 1)  # modo espejo: más natural al jugar
            feat = tracker.process(frame)
            if feat is not None:
                x1, y1, x2, y2 = feat["eyes"]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 160), 2)
                cv2.circle(frame, (x1, y1), 4, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 4, (0, 0, 255), -1)
            # Ajusta al recuadro conservando la proporción (la imagen puede ser vertical)
            fh, fw = frame.shape[:2]
            scale = min(CAM_W / fw, CAM_H / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            x0, y0 = (CAM_W - nw) // 2, (CAM_H - nh) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh))
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            cam_surface = pygame.image.frombuffer(rgb.tobytes(), (CAM_W, CAM_H), "RGB")

        status_text, progress = "", 0.0
        if ctrl.calibrating:
            status_text, progress = ctrl.calibrate(feat, now)
        elif not game.over:
            move, rotate, drop = ctrl.events(feat, now)
            for k in key_events:  # respaldo por teclado
                if k == pygame.K_LEFT:
                    move = -1
                elif k == pygame.K_RIGHT:
                    move = 1
                elif k == pygame.K_UP:
                    rotate = True
                elif k == pygame.K_SPACE:
                    drop = True
            if move:
                game.move(move)
            if rotate:
                game.rotate()
            if drop:
                game.hard_drop()
            game.update(now)

        # ------------------------------ dibujo ------------------------------ #
        screen.fill(BG)
        draw_game(screen, fonts, game)

        if cam_surface:
            screen.blit(cam_surface, (CAM_X, CAM_Y))
        pygame.draw.rect(screen, TEXT, (CAM_X - 2, CAM_Y - 2, CAM_W + 4, CAM_H + 4), 2)
        if feat is None:
            screen.blit(fonts["m"].render("¡Cara no detectada!", True, (255, 90, 90)),
                        (CAM_X + 60, CAM_Y + CAM_H + 8))

        y = CAM_Y + CAM_H + 40
        if feat is not None:
            dr = ctrl.tilt(feat) if (ctrl.calibrating or ctrl.dr_s is None) else ctrl.dr_s
            draw_bar(screen, fonts["s"], f"Inclinación {dr:+.0f}° (mover)",
                     (dr + 30) / 60, (ROLL_ON + 30) / 60, CAM_X, y, CAM_W)
            draw_bar(screen, fonts["s"], "Parpadeo (rotar)", feat["blink"],
                     None if ctrl.calibrating else ctrl.blink_thr, CAM_X, y + 50, CAM_W)
            draw_bar(screen, fonts["s"], "Cejas (caída rápida)", feat["brow"],
                     None if ctrl.calibrating else ctrl.brow_thr, CAM_X, y + 100, CAM_W)

        screen.blit(fonts["m"].render(f"Puntos: {game.score}", True, TEXT), (CAM_X, y + 160))
        screen.blit(fonts["m"].render(f"Líneas: {game.lines}   Nivel: {game.level}", True, TEXT),
                    (CAM_X, y + 190))
        nx, ny = CAM_X + 250, y + 160
        screen.blit(fonts["s"].render("Siguiente", True, TEXT), (nx, ny - 20))
        for r, row in enumerate(SHAPES[game.next_kind]):
            for c, v in enumerate(row):
                if v:
                    draw_block(screen, nx + c * 18, ny + r * 18, COLORS[game.next_kind])
        screen.blit(fonts["s"].render("C: recalibrar   R: reiniciar   ESC: salir", True,
                                      (150, 150, 170)), (CAM_X, WIN_H - 30))

        if ctrl.calibrating:
            overlay = pygame.Surface((game.W * CELL, game.H * CELL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (BOARD_X, BOARD_Y))
            screen.blit(fonts["m"].render(f"Calibración {ctrl.step + 1}/3", True, ACCENT),
                        (BOARD_X + 20, BOARD_Y + 220))
            screen.blit(fonts["s"].render(status_text, True, TEXT), (BOARD_X + 20, BOARD_Y + 260))
            pygame.draw.rect(screen, GRID_LINE, (BOARD_X + 20, BOARD_Y + 290, 260, 14))
            pygame.draw.rect(screen, ACCENT, (BOARD_X + 20, BOARD_Y + 290,
                                              int(260 * progress), 14))
        elif game.over:
            overlay = pygame.Surface((game.W * CELL, game.H * CELL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            screen.blit(overlay, (BOARD_X, BOARD_Y))
            screen.blit(fonts["l"].render("GAME OVER", True, (255, 90, 90)),
                        (BOARD_X + 25, BOARD_Y + 240))
            screen.blit(fonts["s"].render("Pulsa R para reiniciar", True, TEXT),
                        (BOARD_X + 70, BOARD_Y + 300))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
