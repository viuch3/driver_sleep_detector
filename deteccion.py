"""Nucleo de deteccion: malla facial + CNN de ojos + reglas temporales.

Adaptado de realtime_eye_check.py. Cambios respecto al original:

  - El tamano de entrada se lee del propio modelo, no esta cableado, asi
    que funciona igual con el modelo de 64x64 o con uno de otra medida.
  - La logica de alarma vive en esta clase y no en el bucle de la
    interfaz, para que la ventana pueda dibujar sin saber nada de esto.
  - Todo corre en local: no hay red de por medio.
"""

import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:  # Keras 3 independiente
    import keras
except ImportError:  # respaldo: el Keras que trae TensorFlow
    from tensorflow import keras

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "models" / "model_eye.keras"

# Respaldo: el modelo del proyecto anterior (96x96, formato Keras 2).
# Si el .keras nuevo no se puede leer con la version de Keras instalada,
# la app sigue funcionando con este en lugar de no arrancar.
RUTA_MODELO_RESPALDO = RAIZ / "models" / "modelo_ojos.h5"

# Esquinas interna y externa de cada ojo en la malla de MediaPipe.
IDX_OJOS = (("izquierdo", [33, 133]), ("derecho", [362, 263]))

# Margen en pixeles alrededor del ojo, como en el script original.
MARGEN = 50

# Giro maximo de cabeza (grados) para que la alarma sea valida. Si el
# conductor mira de lado, los ojos se ven de perfil, el recorte sale malo
# y el modelo suele leerlos como cerrados. Alarmar ahi seria un falso
# positivo casi seguro.
YAW_MAXIMO = 20.0

# Frames seguidos que deben cumplir la condicion antes de empezar a
# contar. Evita que un unico frame mal clasificado arranque el reloj.
FRAMES_CONFIRMACION = 3

VERDE = (0, 255, 0)
ROJO = (0, 0, 255)
AMARILLO = (0, 200, 255)

# Puntos 3D de referencia de una cara, para estimar el giro con solvePnP.
_PUNTOS_3D = np.array([
    (0.0, 0.0, 0.0),          # nariz
    (0.0, -330.0, -65.0),     # barbilla
    (-225.0, 170.0, -135.0),  # ojo izquierdo
    (225.0, 170.0, -135.0),   # ojo derecho
    (-150.0, -150.0, -125.0), # boca izquierda
    (150.0, -150.0, -125.0),  # boca derecha
])
_IDX_3D = [1, 152, 33, 263, 61, 291]


class DetectorSomnolencia:
    """Procesa frames y decide cuando hay que alarmar."""

    def __init__(self, ruta_modelo: Path = RUTA_MODELO) -> None:
        self.modelo, self.nombre_modelo = self._cargar(ruta_modelo)
        _, alto, ancho, _ = self.modelo.input_shape
        self.tamano = (int(ancho), int(alto))  # (w, h) para cv2.resize

        # Calentamiento: obliga a construir el grafo ahora y no en el
        # primer frame, que si no se nota como un tiron al iniciar.
        self.modelo.predict(
            np.zeros((1, int(alto), int(ancho), 1), dtype="float32"), verbose=0
        )

        self.malla = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cerrados_desde: float | None = None
        self.confirmacion = deque(maxlen=FRAMES_CONFIRMACION)

    @staticmethod
    def _cargar(ruta: Path):
        """Carga el modelo principal; si falla, recurre al de respaldo."""
        candidatos = [Path(ruta), RUTA_MODELO_RESPALDO]
        errores = []
        for candidato in candidatos:
            if not candidato.exists():
                errores.append(f"{candidato.name}: no existe")
                continue
            try:
                return keras.models.load_model(str(candidato)), candidato.name
            except Exception as error:
                errores.append(f"{candidato.name}: {error}")
        raise RuntimeError("No se pudo cargar ningun modelo.\n" + "\n".join(errores))

    # ------------------------------------------------------------------ #

    def procesar(self, frame, umbral: float, segundos_alarma: float) -> tuple:
        """Analiza un frame. Devuelve (frame anotado, info).

        info: rostro, probabilidades, estado, segundos_cerrados, alarma, yaw
        """
        alto, ancho = frame.shape[:2]
        resultado = self.malla.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        info = {
            "rostro": False,
            "probabilidades": [],
            "estado": "sin rostro",
            "segundos_cerrados": 0.0,
            "alarma": False,
            "yaw": None,
            "motivo": None,
        }

        if not resultado.multi_face_landmarks:
            # Sin cara no hay nada que juzgar: se corta cualquier alarma en
            # curso. Es preferible callar a sonar por no ver al conductor.
            self._reiniciar_contadores()
            info["motivo"] = "no se detecta el rostro"
            return frame, info

        puntos = resultado.multi_face_landmarks[0].landmark
        info["rostro"] = True

        probabilidades = []
        for _, indices in IDX_OJOS:
            recorte, caja = self._recortar_ojo(frame, puntos, indices)
            if recorte is None:
                continue
            p = float(self.modelo.predict(recorte, verbose=0)[0][0])
            probabilidades.append(p)

            x, y, w, h = caja
            cv2.rectangle(frame, (x, y), (x + w, y + h), VERDE if p >= umbral else ROJO, 2)
            cv2.putText(frame, f"{p:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        VERDE if p >= umbral else ROJO, 2)

        info["probabilidades"] = probabilidades
        if len(probabilidades) < 2:
            # Con un solo ojo localizado no hay informacion suficiente.
            self._reiniciar_contadores()
            info["motivo"] = "no se localizan ambos ojos"
            return frame, info

        yaw = self._estimar_yaw(puntos, ancho, alto)
        info["yaw"] = yaw
        cabeza_al_frente = yaw is not None and abs(yaw) <= YAW_MAXIMO

        # Basta con que UN ojo este cerrado: al dormirse no se cierran los
        # dos a la vez, y esperar a ambos retrasaria la alarma.
        ojo_cerrado = any(p < umbral for p in probabilidades)
        info["estado"] = "cerrados" if ojo_cerrado else "abiertos"

        # La condicion tiene que sostenerse varios frames seguidos antes
        # de arrancar el reloj: un frame suelto mal clasificado no basta.
        condicion = ojo_cerrado and cabeza_al_frente
        self.confirmacion.append(condicion)
        confirmado = len(self.confirmacion) == self.confirmacion.maxlen and all(self.confirmacion)

        ahora = time.monotonic()
        if confirmado:
            if self.cerrados_desde is None:
                self.cerrados_desde = ahora
            info["segundos_cerrados"] = ahora - self.cerrados_desde
            info["alarma"] = info["segundos_cerrados"] >= segundos_alarma
        else:
            self.cerrados_desde = None

        if not cabeza_al_frente:
            info["motivo"] = "cabeza girada"
        elif not ojo_cerrado:
            info["motivo"] = None

        if not cabeza_al_frente:
            cv2.putText(frame, "cabeza girada", (20, alto - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMARILLO, 2)

        return frame, info

    def reiniciar(self) -> None:
        self._reiniciar_contadores()

    def _reiniciar_contadores(self) -> None:
        self.cerrados_desde = None
        self.confirmacion.clear()

    def cerrar(self) -> None:
        self.malla.close()

    # ------------------------------------------------------------------ #

    def _recortar_ojo(self, frame, puntos, indices):
        """Cuadrado centrado en el ojo, de lado max(w, h) + MARGEN.

        Es la misma geometria del script original: se parte de las dos
        esquinas del ojo y se agranda. El cuadrado importa porque las
        imagenes de entrenamiento lo son; un rectangulo alargado se
        deformaria al redimensionar.
        """
        alto, ancho = frame.shape[:2]
        coords = [(int(puntos[i].x * ancho), int(puntos[i].y * alto)) for i in indices]
        x, y, w_, h_ = cv2.boundingRect(np.array(coords))

        lado = max(w_, h_) + MARGEN
        cx, cy = x + w_ // 2, y + h_ // 2

        x1 = max(cx - lado // 2, 0)
        y1 = max(cy - lado // 2, 0)
        x2 = min(x1 + lado, ancho)
        y2 = min(y1 + lado, alto)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None

        ojo = frame[y1:y2, x1:x2]
        ojo = cv2.cvtColor(ojo, cv2.COLOR_BGR2GRAY)
        ojo = cv2.resize(ojo, self.tamano)
        tensor = (ojo.astype("float32") / 255.0).reshape(1, self.tamano[1], self.tamano[0], 1)

        return tensor, (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _estimar_yaw(puntos, ancho: int, alto: int):
        """Giro lateral de la cabeza en grados, con solvePnP."""
        puntos_2d = np.array(
            [(puntos[i].x * ancho, puntos[i].y * alto) for i in _IDX_3D],
            dtype="double",
        )
        focal = float(ancho)
        matriz = np.array([
            [focal, 0, ancho / 2],
            [0, focal, alto / 2],
            [0, 0, 1],
        ], dtype="double")

        ok, rvec, _ = cv2.solvePnP(
            _PUNTOS_3D, puntos_2d, matriz, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rot, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        return float(np.degrees(np.arctan2(-rot[2, 0], sy)))
