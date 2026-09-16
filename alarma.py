"""Reproduccion de la alarma, en bucle y sin dependencias obligatorias.

Intenta pygame (lo que usaba el script original). Si no esta instalado,
cae a 'afplay', el reproductor que viene con macOS. Asi la app funciona
aunque falte una dependencia, en vez de reventar justo cuando tiene que
avisar de que alguien se esta durmiendo.
"""

import shutil
import subprocess
import threading
from pathlib import Path


class Alarma:
    def __init__(self, ruta: Path) -> None:
        self.ruta = Path(ruta)
        self.sonando = False
        self._motor = None
        self._sonido = None
        self._proceso = None
        self._parar = threading.Event()

        if not self.ruta.exists():
            self._motor = "ninguno"
            return

        try:
            import pygame

            pygame.mixer.init()
            self._sonido = pygame.mixer.Sound(str(self.ruta))
            self._motor = "pygame"
        except Exception:
            self._motor = "afplay" if shutil.which("afplay") else "ninguno"

    @property
    def disponible(self) -> bool:
        return self._motor in ("pygame", "afplay")

    def iniciar(self) -> None:
        if self.sonando or not self.disponible:
            return
        self.sonando = True

        if self._motor == "pygame":
            self._sonido.play(loops=-1)
        else:
            self._parar.clear()
            threading.Thread(target=self._bucle_afplay, daemon=True).start()

    def detener(self) -> None:
        if not self.sonando:
            return
        self.sonando = False

        if self._motor == "pygame":
            self._sonido.stop()
        else:
            self._parar.set()
            if self._proceso and self._proceso.poll() is None:
                self._proceso.terminate()

    def _bucle_afplay(self) -> None:
        """afplay reproduce una vez; el bucle lo relanza hasta que se pare."""
        while not self._parar.is_set():
            self._proceso = subprocess.Popen(
                ["afplay", str(self.ruta)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while self._proceso.poll() is None:
                if self._parar.wait(0.1):
                    self._proceso.terminate()
                    return
