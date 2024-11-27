import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, jsonify
import gdown  # Para descargar el modelo desde Google Drive

app = Flask(__name__)

# URL del archivo .h5 alojado en Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1La0fMs7hHsD2ehDy5bc99yh_5HmGCPWu"
MODEL_PATH = "/tmp/model.h5"  # Ruta temporal en Vercel

try:
    if not os.path.exists(MODEL_PATH):
        print("Descargando el modelo desde Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Modelo descargado correctamente.")
    else:
        print("El modelo ya está disponible.")
except Exception as e:
    print(f"Error descargando el modelo: {e}")

# Cargar el modelo entrenado
try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error cargando el modelo: {e}")

# Diccionario de etiquetas de las clases
labels = {0: "Healthy", 1: "Powdery", 2: "Rust"}

# Función para procesar la imagen y realizar predicciones
def getResult(image_path):
    try:
        img = load_img(image_path, target_size=(225, 225))
        x = img_to_array(img)
        x = x.astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        predictions = model.predict(x)[0]
        print(f"Predicciones: {predictions}")
        return predictions
    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return None

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se encontró ningún archivo"}), 400

    f = request.files["file"]

    try:
        # Guardar la imagen subida
        file_path = os.path.join("/tmp", secure_filename(f.filename))
        f.save(file_path)

        # Obtener las predicciones
        predictions = getResult(file_path)
        if predictions is None:
            return jsonify({"error": "No se pudo procesar la imagen"}), 500

        predicted_label = labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # Eliminar el archivo después de procesar
        os.remove(file_path)

        return jsonify({"prediction": predicted_label, "confidence": confidence})
    except Exception as e:
        print(f"Error procesando la solicitud: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
