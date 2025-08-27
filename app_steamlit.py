import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2, os
from datetime import datetime
import numpy as np

# Charger le modèle (inchangé)
model = YOLO("best.pt")

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

# ----------------- Détection vidéo (inchangé) -----------------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"video_result_{timestamp}.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    return out_path

# ----------------- Détection image (inchangé) -----------------
def predict_image(image):
    image = np.array(image)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model.predict(source=image, conf=0.25, verbose=False)
    annotated_image = results[0].plot()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"image_result_{timestamp}.png")
    cv2.imwrite(out_path, annotated_image)

    return out_path



    # Bandeau + logo

st.title("🧠 Stroke-IA – Détection d'AVC par IA")
st.markdown("""
Prototype d’analyse d’images & vidéos (usage démo, non médical).  
Créé et propulsé par **Badsi Djilali — Ingénieur IA / Deep Learning**
""")

# --- Détection Vidéo ---
st.header("Détection sur vidéo")
video_file = st.file_uploader("Uploader une vidéo (mp4, mov, etc.)", type=["mp4", "mov"])
if video_file and st.button("Analyser la vidéo"):
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    result_path = predict_video("temp_video.mp4")
    st.video(result_path)

# --- Détection Image ---
st.header("Détection sur image")
image_file = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])
if image_file and st.button("Analyser l'image"):
    image = Image.open(image_file)
    result_path = predict_image(image)
    st.image(result_path, caption="Image annotée", use_column_width=True)

st.markdown("""
---
⚠️ **Disclaimer :** Stroke-IA est une démonstration technique. Les résultats ne constituent pas un avis médical.
© {year} — Badsi Djilali. Tous droits réservés.
""".format(year=datetime.now().year))



