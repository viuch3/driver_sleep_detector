from contextlib import asynccontextmanager
from pathlib import Path

import keras
import numpy as np
from fastapi import FastAPI, Request

ruta_modelo = Path(__file__).parent / "models" / "model_eye.keras"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- arranque: corre una vez, antes de aceptar peticiones ---
    model = keras.saving.load_model(ruta_modelo)
    model.predict(np.zeros((1, 64, 64, 1), dtype="float32"), verbose=0)
    app.state.model = model

    yield  # aquí la app queda viva atendiendo peticiones

    # --- apagado: corre una vez, al hacer Ctrl+C ---
    app.state.model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
    }
