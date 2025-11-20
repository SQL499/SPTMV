# SPTMV/GUI/app.py

from pathlib import Path
import json

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="IA para salud animal", layout="centered")
API_URL = "http://localhost:8000"

# 0/1 -> etiquetas legibles para el usuario
ETIQUETAS_DERM = {
    0: "Baja probabilidad de enfermedad dermatológica",
    1: "Alta probabilidad de enfermedad dermatológica",
}

@st.cache_data
def cargar_features_clinicas():
    ruta_base = Path(__file__).resolve().parents[1]
    ruta_diccionario = ruta_base / "datasets" / "diccionario_variables_clinicas.xlsx"
    df_dicc = pd.read_excel(ruta_diccionario)
    features = df_dicc["Feature"].tolist()
    return [f for f in features if f != "target_derm"]


FEATURES_X = cargar_features_clinicas()

RAZAS_OPCIONES = [
    "beagle",
    "boxer",
    "bulldog",
    "german shepherd",
    "golden retriever",
    "labrador retriever",
    "mixed breed",
    "poodle",
    "rottweiler",
    "yorkshire terrier",
    "otra",
]


def construir_vector_clinico_desde_formulario(
    edad: float,
    peso: float,
    raza: str,
    fiebre: bool,
    diarrea: bool,
    vomitos: bool,
    prurito: bool,
    erupcion_piel: bool,
    letargo: bool,
    perdida_apetito: bool,
    sintomas_respiratorios: bool,
    historial_piel: bool,
    vacunado: bool,
    enfermedad_cronica: bool,
    alergias: bool,
    enfermedad_renal: bool,
) -> dict:
    # todas las features conocidas a 0
    caracteristicas = {f: 0.0 for f in FEATURES_X}

    # --- básicas ---
    caracteristicas["Age"] = float(edad)
    caracteristicas["Weight_kg"] = float(peso)
    caracteristicas["age_squared"] = float(edad) ** 2
    caracteristicas["age_weight_interaction"] = float(edad) * float(peso)

    if edad < 1:
        caracteristicas["age_bucket_puppy"] = 1.0
    elif edad < 7:
        caracteristicas["age_bucket_adult"] = 1.0
    else:
        caracteristicas["age_bucket_senior"] = 1.0

    # --- raza agrupada ---
    raza = raza.lower()
    if raza == "otra":
        caracteristicas["breed_is_other"] = 1.0
    else:
        col_raza = f"breed_grouped_{raza}"
        if col_raza in caracteristicas:
            caracteristicas[col_raza] = 1.0

    # --- historial / condiciones crónicas ---
    caracteristicas["is_vaccinated"] = 1.0 if vacunado else 0.0
    caracteristicas["has_chronic_illness"] = 1.0 if enfermedad_cronica else 0.0
    caracteristicas["has_allergies"] = 1.0 if alergias else 0.0
    caracteristicas["has_kidney_history"] = 1.0 if enfermedad_renal else 0.0
    caracteristicas["has_skin_history"] = 1.0 if historial_piel else 0.0

    caracteristicas["chronic_conditions_count"] = float(
        caracteristicas["has_chronic_illness"]
        + caracteristicas["has_allergies"]
        + caracteristicas["has_kidney_history"]
    )

    # --- síntomas individuales ---
    has_fever = 1.0 if fiebre else 0.0
    has_diarrhea = 1.0 if diarrea else 0.0
    has_vomiting = 1.0 if vomitos else 0.0
    has_lethargy = 1.0 if letargo else 0.0
    has_skin_rashes = 1.0 if erupcion_piel else 0.0
    has_poor_appetite = 1.0 if perdida_apetito else 0.0  # usa el nombre exacto de la columna

    caracteristicas["has_fever"] = has_fever
    caracteristicas["has_diarrhea"] = has_diarrhea
    caracteristicas["has_vomiting"] = has_vomiting
    caracteristicas["has_lethargy"] = has_lethargy
    caracteristicas["has_skin_rashes"] = has_skin_rashes
    caracteristicas["has_poor_appetite"] = has_poor_appetite
    caracteristicas["has_loss_of_appetite"] = has_poor_appetite


    # --- contadores de síntomas ---
    caracteristicas["gi_symptoms_count"] = float(
        has_diarrhea + has_vomiting + has_poor_appetite
    )

    caracteristicas["systemic_symptoms_count"] = float(
        has_fever + has_lethargy
    )

    caracteristicas["respiratory_symptoms_count"] = 1.0 if sintomas_respiratorios else 0.0

    caracteristicas["dermatological_symptoms_count"] = float(
        (1.0 if prurito else 0.0) + has_skin_rashes
    )

    caracteristicas["total_symptoms_count"] = float(
        has_fever
        + has_diarrhea
        + has_vomiting
        + has_lethargy
        + has_poor_appetite
        + caracteristicas["respiratory_symptoms_count"]
        + caracteristicas["dermatological_symptoms_count"]
    )

    return caracteristicas



def mostrar_resultado_clasificacion(
    datos_respuesta: dict, titulo: str = "Resultado", etiquetas: dict | None = None
):
    clases = datos_respuesta.get("clases", [])
    probs = datos_respuesta.get("probabilidades", [])
    pred = datos_respuesta.get("prediccion", None)

    # Si no viene pred explícito, la calculamos
    if pred is None and clases and probs:
        indice_pred = int(np.argmax(probs))
        pred = clases[indice_pred]

    st.subheader(titulo)

    # Texto bonito para la predicción
    if pred is not None:
        pred_num = pred
        try:
            pred_num = int(pred)
        except Exception:
            pass

        if etiquetas and isinstance(pred_num, int) and pred_num in etiquetas:
            texto = etiquetas[pred_num]
            st.write(f"Predicción: **{texto}** (clase {pred_num})")
        else:
            st.write(f"Predicción: **{pred}**")

    # Probabilidades en formato legible
    if clases and probs:
        st.write("Probabilidades:")
        for c, p in zip(clases, probs):
            c_mostrada = c
            try:
                c_int = int(c)
                if etiquetas and c_int in etiquetas:
                    c_mostrada = f"{c_int} ({etiquetas[c_int]})"
            except Exception:
                pass
            st.write(f"- {c_mostrada}: {p:.3f}")



def main():
    st.title("Plataforma IA para salud animal")
    st.caption("Demo académica: modelo clínico, modelo de imágenes y modelo multimodal")

    pestañas = st.tabs(["Modelo clínico", "Modelo por imágenes", "Modelo multimodal"])

    # --- Pestaña 1: modelo clínico ---
    with pestañas[0]:
        st.header("Predicción basada en datos clínicos")
        with st.form("form_clinico"):
            col1, col2 = st.columns(2)
            with col1:
                edad = st.number_input("Edad (años)", min_value=0.0, value=3.0, step=0.5)
                peso = st.number_input(
                    "Peso (kg)", min_value=0.0, value=12.0, step=0.5
                )
                raza = st.selectbox("Raza agrupada", RAZAS_OPCIONES)
            with col2:
                st.markdown("**Historial médico**")
                vacunado = st.checkbox("Vacunado al día")
                enfermedad_cronica = st.checkbox("Otra enfermedad crónica conocida")
                alergias = st.checkbox("Alergias diagnosticadas")
                enfermedad_renal = st.checkbox("Antecedente de enfermedad renal")
                historial_piel = st.checkbox("Antecedente de problemas de piel")

                st.markdown("**Síntomas actuales**")
                fiebre = st.checkbox("Fiebre")
                diarrea = st.checkbox("Diarrea")
                vomitos = st.checkbox("Vómitos")
                perdida_apetito = st.checkbox("Pérdida de apetito / come menos")
                prurito = st.checkbox("Prurito / rascado")
                erupcion_piel = st.checkbox("Lesiones o erupciones en la piel")
                sintomas_respiratorios = st.checkbox(
                    "Síntomas respiratorios (tos, estornudos, dificultad respiratoria)"
                )
                letargo = st.checkbox("Letargia / baja energía")


            enviado = st.form_submit_button("Predecir con modelo clínico")

        if enviado:
            caracteristicas = construir_vector_clinico_desde_formulario(
                edad=edad,
                peso=peso,
                raza=raza,
                fiebre=fiebre,
                diarrea=diarrea,
                vomitos=vomitos,
                prurito=prurito,
                erupcion_piel=erupcion_piel,
                letargo=letargo,
                perdida_apetito=perdida_apetito,
                sintomas_respiratorios=sintomas_respiratorios,
                historial_piel=historial_piel,
                vacunado=vacunado,
                enfermedad_cronica=enfermedad_cronica,
                alergias=alergias,
                enfermedad_renal=enfermedad_renal,
            )


            try:
                respuesta = requests.post(
                    f"{API_URL}/predict_clinico",
                    json={"caracteristicas": caracteristicas},
                    timeout=10,
                )
                if respuesta.ok:
                    mostrar_resultado_clasificacion(
                        respuesta.json(),
                        titulo="Resultado modelo clínico",
                        etiquetas=ETIQUETAS_DERM,
                    )
                else:
                    st.error(f"Error en la API: {respuesta.status_code}")
            except Exception as e:
                st.error(f"No se pudo conectar con la API: {e}")

    # --- Pestaña 2: modelo de imágenes ---
    with pestañas[1]:
        st.header("Predicción basada en imagen dermatológica")
        imagen_subida = st.file_uploader(
            "Sube una imagen de la lesión de piel",
            type=["jpg", "jpeg", "png"],
        )
        if st.button("Predecir con modelo de imágenes"):
            if imagen_subida is None:
                st.warning("Primero sube una imagen.")
            else:
                try:
                    files = {
                        "imagen": (
                            imagen_subida.name,
                            imagen_subida.getvalue(),
                            imagen_subida.type,
                        )
                    }
                    respuesta = requests.post(
                        f"{API_URL}/predict_imagen", files=files, timeout=20
                    )
                    if respuesta.ok:
                        datos = respuesta.json()
                        mostrar_resultado_clasificacion(
                            datos, titulo="Resultado modelo de imágenes"
                        )
                    else:
                        st.error(f"Error en la API: {respuesta.status_code}")
                except Exception as e:
                    st.error(f"No se pudo conectar con la API: {e}")

    # --- Pestaña 3: modelo multimodal ---
    with pestañas[2]:
        st.header("Predicción multimodal (clínico + imagen)")
        st.markdown(
            "Usa los mismos datos clínicos del formulario y una imagen para combinar ambos modelos."
        )

        with st.form("form_multimodal"):
            col1, col2 = st.columns(2)
            with col1:
                edad_m = st.number_input(
                    "Edad (años)", min_value=0.0, value=3.0, step=0.5, key="edad_m"
                )
                peso_m = st.number_input(
                    "Peso (kg)", min_value=0.0, value=12.0, step=0.5, key="peso_m"
                )
                raza_m = st.selectbox(
                    "Raza agrupada", RAZAS_OPCIONES, key="raza_m"
                )
            with col2:
                st.markdown("**Historial médico**")
                vacunado_m = st.checkbox("Vacunado al día", key="vacunado_m")
                enfermedad_cronica_m = st.checkbox(
                    "Otra enfermedad crónica conocida", key="enf_cronica_m"
                )
                alergias_m = st.checkbox(
                    "Alergias diagnosticadas", key="alergias_m"
                )
                enfermedad_renal_m = st.checkbox(
                    "Antecedente de enfermedad renal", key="enf_renal_m"
                )
                historial_piel_m = st.checkbox(
                    "Antecedente de problemas de piel", key="hist_piel_m"
                )

                st.markdown("**Síntomas actuales**")
                fiebre_m = st.checkbox("Fiebre", key="fiebre_m")
                diarrea_m = st.checkbox("Diarrea", key="diarrea_m")
                vomitos_m = st.checkbox("Vómitos", key="vomitos_m")
                perdida_apetito_m = st.checkbox(
                    "Pérdida de apetito / come menos", key="perdida_apetito_m"
                )
                prurito_m = st.checkbox(
                    "Prurito / rascado", key="prurito_m"
                )
                erupcion_piel_m = st.checkbox(
                    "Lesiones o erupciones en la piel", key="erupcion_piel_m"
                )
                sintomas_respiratorios_m = st.checkbox(
                    "Síntomas respiratorios (tos, estornudos, dificultad respiratoria)",
                    key="sint_resp_m",
                )
                letargo_m = st.checkbox(
                    "Letargia / baja energía", key="letargo_m"
                )

            imagen_m = st.file_uploader(
                "Sube una imagen de la lesión de piel",
                type=["jpg", "jpeg", "png"],
                key="imagen_m",
            )

            enviado_m = st.form_submit_button("Predecir con modelo multimodal")

        if enviado_m:
            if imagen_m is None:
                st.warning("Primero sube una imagen.")
            else:
                caracteristicas_m = construir_vector_clinico_desde_formulario(
                    edad=edad_m,
                    peso=peso_m,
                    raza=raza_m,
                    fiebre=fiebre_m,
                    diarrea=diarrea_m,
                    vomitos=vomitos_m,
                    prurito=prurito_m,
                    erupcion_piel=erupcion_piel_m,
                    letargo=letargo_m,
                    perdida_apetito=perdida_apetito_m,
                    sintomas_respiratorios=sintomas_respiratorios_m,
                    historial_piel=historial_piel_m,
                    vacunado=vacunado_m,
                    enfermedad_cronica=enfermedad_cronica_m,
                    alergias=alergias_m,
                    enfermedad_renal=enfermedad_renal_m,
                )
                try:
                    files = {
                        "imagen": (
                            imagen_m.name,
                            imagen_m.getvalue(),
                            imagen_m.type,
                        )
                    }
                    data = {"datos_clinicos": json.dumps(caracteristicas_m)}
                    respuesta = requests.post(
                        f"{API_URL}/predict_multimodal",
                        data=data,
                        files=files,
                        timeout=30,
                    )
                    if respuesta.ok:
                        datos = respuesta.json()
                        pred_multi = datos.get("prediccion_multimodal")
                        st.subheader("Resultado multimodal")
                        if pred_multi is not None:
                            try:
                                pred_multi_int = int(pred_multi)
                            except Exception:
                                pred_multi_int = pred_multi

                            if isinstance(pred_multi_int, int) and pred_multi_int in ETIQUETAS_DERM:
                                texto_multi = ETIQUETAS_DERM[pred_multi_int]
                                st.write(
                                    f"Predicción final (multimodal): **{texto_multi}** (clase {pred_multi_int})"
                                )
                            else:
                                st.write(f"Predicción final (multimodal): **{pred_multi}**")

                        st.markdown("**Detalle por módulo:**")

                        st.write("- Modelo clínico:")
                        salida_clinico = datos.get("salida_clinico", {})
                        mostrar_resultado_clasificacion(
                            salida_clinico,
                            titulo="Modelo clínico",
                            etiquetas=ETIQUETAS_DERM
                        )

                        st.write("- Modelo de imágenes:")
                        salida_imagen = datos.get("salida_imagen", {})
                        mostrar_resultado_clasificacion(
                            salida_imagen, titulo="Modelo de imágenes"
                        )
                    else:
                        st.error(f"Error en la API: {respuesta.status_code}")
                except Exception as e:
                    st.error(f"No se pudo conectar con la API: {e}")



if __name__ == "__main__":
    main()
