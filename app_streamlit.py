import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Detección de Dengue por Dron", page_icon="🦟", layout="centered")

st.title("🦟 Detección de Criaderos de Dengue con Dron e IA")
st.write("Sube una imagen aérea tomada por dron o ingresa un enlace web para evaluar si la zona presenta riesgo de criaderos.")

@st.cache_resource
def cargar_modelo():
    modelo_path = "modelo_dengue_transfer.keras"
    if os.path.exists(modelo_path):
        return load_model(modelo_path, compile=False)
    return None

modelo = cargar_modelo()
class_names = ['Con_Riesgo', 'Sin_Riesgo']

if modelo is None:
    st.error("⚠️ No se encontró el archivo 'modelo_dengue_transfer.keras' en el directorio. Asegúrate de incluirlo.")
else:
    opcion = st.radio("Selecciona el método de entrada:", ("Subir imagen desde el equipo", "Ingresar enlace de imagen (URL)"))
    
    imagen_cargada = None

    if opcion == "Subir imagen desde el equipo":
        archivo_subido = st.file_uploader("Elige una imagen (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        if archivo_subido is not None:
            imagen_cargada = Image.open(archivo_subido)
    else:
        url_imagen = st.text_input("Ingresa la URL de la imagen:")
        if url_imagen:
            try:
                response = requests.get(url_imagen)
                imagen_cargada = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"No se pudo cargar la imagen desde el enlace: {e}")

    if imagen_cargada is not None:
        st.image(imagen_cargada, caption="Imagen seleccionada para análisis", use_container_width=True)
        
        if st.button("🔍 Analizar Imagen"):
            with st.spinner("Analizando con Inteligencia Artificial..."):
                # Preprocesar imagen
                img_resized = imagen_cargada.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = tf.expand_dims(img_array, 0)
                
                # Predicción
                prediccion = modelo.predict(img_array)
                score = tf.nn.softmax(prediccion[0])
                
                clase_predicha = class_names[np.argmax(score)]
                certeza = 100 * np.max(score)
                
                st.markdown("---")
                st.subheader("📊 Resultado del Diagnóstico")
                
                if clase_predicha == 'Con_Riesgo':
                    st.error(f"🔴 **Alerta: Con Riesgo** (Confianza: {certeza:.2f}%)")
                    st.write("Se han detectado patrones visuales asociados a acumulación de agua estancada, pozos o canales hídricos descubiertos.")
                else:
                    st.success(f"🟢 **Zona Segura / Sin Riesgo** (Confianza: {certeza:.2f}%)")
                    st.write("El terreno muestra características secas, caminos limpios o vegetación sin indicios críticos de anegamiento.")
