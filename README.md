# 🦟 Detección de Criaderos de Dengue con Dron e IA

Este proyecto implementa un modelo de Visión Artificial utilizando **Transfer Learning** (`MobileNetV2` en TensorFlow/Keras) para clasificar imágenes aéreas tomadas por drones en zonas rurales (`Con_Riesgo` vs `Sin_Riesgo`).

## 🚀 Contenido del Proyecto
- `app_streamlit.py`: Aplicación web interactiva para subir imágenes o ingresar enlaces y obtener la predicción en tiempo real.
- `modelo_dengue_transfer.keras`: El modelo entrenado.
- `requirements.txt`: Dependencias necesarias para ejecutar la aplicación.

## 🌐 Cómo ejecutar la aplicación web localmente
1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta la aplicación con Streamlit:
   ```bash
   streamlit run app_streamlit.py
   ```
