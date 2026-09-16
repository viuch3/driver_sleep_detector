"""Detector de somnolencia al volante - aplicacion de escritorio.

Ejecutar:
    python app.py

Todo corre en local: camara, modelo y alarma. No necesita internet, que
es justo lo que hace falta en carretera.

Arquitectura: un hilo de trabajo captura y analiza frames; la ventana
solo dibuja. Tkinter no es seguro para uso desde varios hilos, asi que el
hilo nunca toca la interfaz: deja el ultimo frame en una variable y la
ventana lo recoge con after().
"""

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from alarma import Alarma
from deteccion import DetectorSomnolencia

RAIZ = Path(__file__).resolve().parent
RUTA_ALARMA = RAIZ / "assets" / "alarm.wav"

ANCHO_VISTA = 640
ALTO_VISTA = 480

FONDO = "#12141a"
PANEL = "#1c1f28"
TEXTO = "#e8eaf0"
TENUE = "#8a90a3"
OK = "#2ecc71"
PELIGRO = "#e74c3c"


class Aplicacion:
    def __init__(self, raiz: tk.Tk) -> None:
        self.raiz = raiz
        self.raiz.title("Detector de somnolencia")
        self.raiz.configure(bg=FONDO)
        self.raiz.protocol("WM_DELETE_WINDOW", self.al_cerrar)

        self.detector: DetectorSomnolencia | None = None
        self.alarma = Alarma(RUTA_ALARMA)
        self.camara = None
        self.hilo: threading.Thread | None = None
        self.corriendo = False

        self._lock = threading.Lock()
        self._ultimo_frame = None
        self._ultima_info = None
        self._marcas = []

        self._construir_interfaz()
        self._refrescar()

    # ------------------------------------------------------------ interfaz

    def _construir_interfaz(self) -> None:
        contenedor = tk.Frame(self.raiz, bg=FONDO, padx=16, pady=16)
        contenedor.pack(fill="both", expand=True)

        tk.Label(contenedor, text="Detector de somnolencia", bg=FONDO, fg=TEXTO,
                 font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(contenedor, text="Verde: ojos abiertos    Rojo: ojos cerrados",
                 bg=FONDO, fg=TENUE, font=("Helvetica", 11)).pack(anchor="w", pady=(0, 12))

        # Ojo: en un Label, width/height se miden EN CARACTERES, no en
        # pixeles. Por eso el area de video va dentro de un Frame, donde
        # si son pixeles, con pack_propagate(False) para que no encoja al
        # tamano de su contenido.
        marco_video = tk.Frame(contenedor, bg="black", width=ANCHO_VISTA, height=ALTO_VISTA)
        marco_video.pack()
        marco_video.pack_propagate(False)

        self.vista = tk.Label(marco_video, bg="black")
        self.vista.pack(expand=True)

        self.aviso = tk.Label(contenedor, text="", bg=PELIGRO, fg="white",
                              font=("Helvetica", 16, "bold"), pady=8)

        botones = tk.Frame(contenedor, bg=FONDO)
        botones.pack(fill="x", pady=12)

        self.btn_iniciar = tk.Button(botones, text="Iniciar", command=self.iniciar,
                                     font=("Helvetica", 13, "bold"), width=12,
                                     highlightbackground=FONDO)
        self.btn_iniciar.pack(side="left")

        self.btn_detener = tk.Button(botones, text="Detener", command=self.detener,
                                     font=("Helvetica", 13), width=12,
                                     state="disabled", highlightbackground=FONDO)
        self.btn_detener.pack(side="left", padx=8)

        ajustes = tk.Frame(contenedor, bg=PANEL, padx=14, pady=12)
        ajustes.pack(fill="x")

        # El umbral se deja ajustable a proposito: el modelo satura cerca
        # de 1.0 con imagenes de webcam, asi que el corte util no es 0.5
        # y conviene poder afinarlo en vivo viendo las probabilidades.
        tk.Label(ajustes, text="Umbral de ojo abierto", bg=PANEL, fg=TENUE,
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        self.umbral = tk.DoubleVar(value=0.99)
        tk.Scale(ajustes, from_=0.50, to=0.999, resolution=0.005,
                 orient="horizontal", variable=self.umbral, length=260,
                 bg=PANEL, fg=TEXTO, highlightthickness=0, troughcolor=FONDO
                 ).grid(row=1, column=0, padx=(0, 24))

        # macOS suele ofrecer varias camaras (la del Mac, el iPhone por
        # Continuity, capturadoras). El indice 0 no siempre es la del
        # equipo, asi que se deja elegir.
        tk.Label(ajustes, text="Camara", bg=PANEL, fg=TENUE,
                 font=("Helvetica", 10)).grid(row=0, column=2, sticky="w", padx=(24, 0))
        self.indice_camara = tk.IntVar(value=0)
        self.selector_camara = tk.OptionMenu(ajustes, self.indice_camara, 0, 1, 2, 3)
        self.selector_camara.config(bg=PANEL, fg=TEXTO, highlightthickness=0, width=4)
        self.selector_camara.grid(row=1, column=2, sticky="w", padx=(24, 0))

        tk.Label(ajustes, text="Segundos para la alarma", bg=PANEL, fg=TENUE,
                 font=("Helvetica", 10)).grid(row=0, column=1, sticky="w")
        self.segundos = tk.DoubleVar(value=1.2)
        tk.Scale(ajustes, from_=0.3, to=4.0, resolution=0.1,
                 orient="horizontal", variable=self.segundos, length=260,
                 bg=PANEL, fg=TEXTO, highlightthickness=0, troughcolor=FONDO
                 ).grid(row=1, column=1)

        self.estado = tk.Label(contenedor, text="Listo.", bg=FONDO, fg=TENUE,
                               font=("Helvetica", 11), anchor="w")
        self.estado.pack(fill="x", pady=(10, 0))

        if not self.alarma.disponible:
            self.estado.config(
                text=f"Aviso: no se pudo preparar el sonido ({RUTA_ALARMA.name}). "
                     "La deteccion funciona, pero no habra alarma audible."
            )

    # -------------------------------------------------------------- acciones

    def iniciar(self) -> None:
        self.btn_iniciar.config(state="disabled")
        self.estado.config(text="Cargando modelo y abriendo camara...")
        self.raiz.update_idletasks()

        try:
            if self.detector is None:
                self.detector = DetectorSomnolencia()
            self.detector.reiniciar()

            indice = self.indice_camara.get()
            # AVFoundation es el backend nativo de macOS: abrir sin
            # especificarlo a veces engancha una camara distinta.
            self.camara = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
            if not self.camara.isOpened():
                self.camara = cv2.VideoCapture(indice)
            if not self.camara.isOpened():
                raise RuntimeError(
                    f"No se pudo abrir la camara {indice}. Prueba otro indice en el "
                    "selector, o revisa los permisos de camara para la terminal "
                    "en Ajustes del sistema > Privacidad y seguridad."
                )
            self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception as error:
            self.estado.config(text=f"Error: {error}")
            self.btn_iniciar.config(state="normal")
            if self.camara is not None:
                self.camara.release()
                self.camara = None
            return

        self.corriendo = True
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

        self.btn_detener.config(state="normal")
        self.selector_camara.config(state="disabled")
        self.estado.config(text=f"En marcha. Modelo: {self.detector.nombre_modelo}")

    def detener(self) -> None:
        self.corriendo = False
        if self.hilo is not None:
            self.hilo.join(timeout=2.0)
            self.hilo = None
        if self.camara is not None:
            self.camara.release()
            self.camara = None

        self.alarma.detener()
        self.aviso.pack_forget()
        with self._lock:
            self._ultimo_frame = None
            self._ultima_info = None
        self.vista.config(image="")
        self.vista.image = None

        self.btn_iniciar.config(state="normal")
        self.btn_detener.config(state="disabled")
        self.selector_camara.config(state="normal")
        self.estado.config(text="Detenido.")

    def al_cerrar(self) -> None:
        self.detener()
        if self.detector is not None:
            self.detector.cerrar()
        self.raiz.destroy()

    # ----------------------------------------------------------- hilo y UI

    def _bucle(self) -> None:
        """Hilo de trabajo: captura, analiza y guarda el resultado."""
        while self.corriendo:
            ok, frame = self.camara.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # espejo, mas natural para el usuario
            frame, info = self.detector.procesar(
                frame, self.umbral.get(), self.segundos.get()
            )

            with self._lock:
                self._ultimo_frame = frame
                self._ultima_info = info

    def _refrescar(self) -> None:
        """Se ejecuta en el hilo principal; es el unico que toca Tkinter."""
        with self._lock:
            frame = self._ultimo_frame
            info = self._ultima_info

        if frame is not None:
            self._pintar_frame(frame)
        if info is not None:
            self._pintar_estado(info)

        self.raiz.after(30, self._refrescar)

    def _pintar_frame(self, frame) -> None:
        alto, ancho = frame.shape[:2]
        escala = min(ANCHO_VISTA / ancho, ALTO_VISTA / alto)
        vista = cv2.resize(frame, (int(ancho * escala), int(alto * escala)))
        imagen = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(vista, cv2.COLOR_BGR2RGB)))

        self.vista.config(image=imagen)
        self.vista.image = imagen  # sin esta referencia, el recolector la borra

        ahora = time.monotonic()
        self._marcas = [t for t in self._marcas if ahora - t < 1.0]
        self._marcas.append(ahora)

    def _pintar_estado(self, info: dict) -> None:
        if info["alarma"]:
            self.alarma.iniciar()
            self.aviso.config(text="DESPIERTA")
            if not self.aviso.winfo_ismapped():
                self.aviso.pack(fill="x", pady=(8, 0))
        else:
            self.alarma.detener()
            if self.aviso.winfo_ismapped():
                self.aviso.pack_forget()

        probs = "  ".join(f"{p:.2f}" for p in info["probabilidades"]) or "-"
        yaw = f"{info['yaw']:.0f}" if info["yaw"] is not None else "-"
        motivo = f"    [alarma inhibida: {info['motivo']}]" if info.get("motivo") else ""
        self.estado.config(
            text=(
                f"Estado: {info['estado']}    "
                f"Probabilidades: {probs}    "
                f"Cerrados: {info['segundos_cerrados']:.1f} s    "
                f"Giro: {yaw}    "
                f"FPS: {len(self._marcas)}{motivo}"
            ),
            fg=PELIGRO if info["estado"] == "cerrados" else TENUE,
        )


def main() -> None:
    raiz = tk.Tk()
    Aplicacion(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
