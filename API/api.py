# SPTMV/API/main.py

from pathlib import Path
from typing import Dict

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input
except ImportError:
    from keras.models import load_model
    from keras.applications.efficientnet import preprocess_input


# --- Rutas base del proyecto ---

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_MODELOS = RUTA_PROYECTO / "Modelos Guardados"
RUTA_DATASETS = RUTA_PROYECTO / "datasets"


# --- Carga de features clínicas ---

def cargar_features_clinicas():
    ruta_diccionario = RUTA_DATASETS / "diccionario_variables_clinicas.xlsx"
    df_dicc = pd.read_excel(ruta_diccionario)
    features = df_dicc["Feature"].tolist()
    features_x = [f for f in features if f != "target_derm"]
    return features_x


# --- Carga de modelos y clases de la CNN ---

def cargar_modelos_y_clases():
    modelo_clinico = joblib.load(RUTA_MODELOS / "modelo_rf_clinico.joblib")
    modelo_multimodal = joblib.load(RUTA_MODELOS / "modelo_multimodal.joblib")
    modelo_imagenes = load_model(RUTA_MODELOS / "modelo_cnn_piel_perros.keras")

    ruta_clases = RUTA_MODELOS / "clases_cnn_imagenes.json"
    with open(ruta_clases, "r", encoding="utf-8") as f:
        dicc_clases = json.load(f)

    # dicc_clases tiene:
    # - "clases_a_indices": {nombre_clase: idx}
    # - "indices_a_clases": {idx_str: nombre_clase}
    dicc_indices_a_clases = dicc_clases.get("indices_a_clases", {})
    mapa_indices_a_clases = {int(k): v for k, v in dicc_indices_a_clases.items()}

    return modelo_clinico, modelo_imagenes, modelo_multimodal, mapa_indices_a_clases


FEATURES_X = cargar_features_clinicas()
(
    MODELO_CLINICO,
    MODELO_IMAGENES,
    MODELO_MULTIMODAL,
    INDICE_A_CLASE_CNN,
) = cargar_modelos_y_clases()

# índice de la clase "1" (caso dermatológico) en el modelo clínico
INDICE_CLASE_DERM = int(np.where(MODELO_CLINICO.classes_ == 1)[0][0])

# lista ordenada de clases de la CNN (posición i = índice de salida i)
CLASES_CNN_ORDENADAS = [INDICE_A_CLASE_CNN[i] for i in sorted(INDICE_A_CLASE_CNN.keys())]


# --- Utilidades internas ---

def preparar_dataframe_clinico(caracteristicas: Dict[str, float]) -> pd.DataFrame:
    fila = {nombre: caracteristicas.get(nombre, 0.0) for nombre in FEATURES_X}
    return pd.DataFrame([fila])


def preprocesar_imagen_bytes(contenido: bytes) -> np.ndarray:
    imagen = Image.open(BytesIO(contenido)).convert("RGB")
    imagen = imagen.resize((224, 224))
    arr = np.asarray(imagen).astype("float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# --- Esquemas y API ---

class EntradaClinica(BaseModel):
    caracteristicas: Dict[str, float]


app = FastAPI(title="API IA Salud Animal", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_clinico")
def predict_clinico(entrada: EntradaClinica):
    df = preparar_dataframe_clinico(entrada.caracteristicas)
    probabilidades = MODELO_CLINICO.predict_proba(df)[0]
    clases = MODELO_CLINICO.classes_.tolist()
    indice_pred = int(np.argmax(probabilidades))
    return {
        "clases": clases,
        "probabilidades": probabilidades.tolist(),
        "prediccion": int(clases[indice_pred]),
    }


@app.post("/predict_imagen")
async def predict_imagen(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    lote = preprocesar_imagen_bytes(contenido)
    probabilidades = MODELO_IMAGENES.predict(lote)[0]
    clases = CLASES_CNN_ORDENADAS
    indice_pred = int(np.argmax(probabilidades))
    return {
        "clases": clases,
        "probabilidades": probabilidades.tolist(),
        "prediccion": clases[indice_pred],
    }


@app.post("/predict_multimodal")
async def predict_multimodal(
    datos_clinicos: str = Form(...),
    imagen: UploadFile = File(...),
):
    # 1) modelo clínico
    caracteristicas = json.loads(datos_clinicos)
    df = preparar_dataframe_clinico(caracteristicas)
    proba_clinico = MODELO_CLINICO.predict_proba(df)[0]
    p_derm = float(proba_clinico[INDICE_CLASE_DERM])

    # 2) modelo de imágenes
    contenido = await imagen.read()
    lote = preprocesar_imagen_bytes(contenido)
    proba_imagen = MODELO_IMAGENES.predict(lote)[0]

    # 3) meta-modelo (stacking)
    vector_meta = np.concatenate([[p_derm], proba_imagen])
    proba_multimodal = MODELO_MULTIMODAL.predict_proba(vector_meta.reshape(1, -1))[0]
    clases_multimodal = MODELO_MULTIMODAL.classes_.tolist()
    indice_pred = int(np.argmax(proba_multimodal))

    return {
        "prediccion_multimodal": int(clases_multimodal[indice_pred]),
        "clases_multimodal": clases_multimodal,
        "probabilidades_multimodal": proba_multimodal.tolist(),
        "salida_clinico": {
            "clases": MODELO_CLINICO.classes_.tolist(),
            "probabilidades": proba_clinico.tolist(),
        },
        "salida_imagen": {
            "clases": CLASES_CNN_ORDENADAS,
            "probabilidades": proba_imagen.tolist(),
        },
    }
