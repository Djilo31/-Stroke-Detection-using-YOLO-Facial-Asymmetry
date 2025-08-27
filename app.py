import gradio as gr
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

# ----------------- Interface Gradio (mise en forme uniquement) -----------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    # Bandeau + logo SVG inline (pas d'appel internet)
    gr.HTML("""
    <div style="display:flex;align-items:center;gap:14px;padding:12px 0 4px;">
      <svg width="44" height="44" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"
           style="flex:0 0 44px">
        <rect x="2" y="2" width="60" height="60" rx="14" fill="#eaf3ff" stroke="#2b6cb0" stroke-width="2"/>
        <path d="M32 14 v36 M14 32 h36" stroke="#2b6cb0" stroke-width="6" stroke-linecap="round"/>
      </svg>
      <div>
        <h1 style="margin:0;font-size:28px;">🧠 Stroke-IA – Détection d'AVC par IA</h1>
        <div style="margin-top:2px;color:#2b6cb0;font-weight:600;">
          Créé et propulsé par <span style="text-decoration:underline;">Badsi Djilali</span> — Ingénieur IA / Deep Learning
        </div>
        <div style="margin-top:4px;color:#4a5568;">Prototype d’analyse d’images & vidéos (usage démo, non médical).</div>
      </div>
    </div>
    <hr style="border:none;border-top:1px solid #e2e8f0;margin:10px 0 14px;">
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Détection sur **vidéo**")
            video_input = gr.Video(label="Uploader une vidéo (mp4, mov, etc.)")
            video_output = gr.Video(label="Vidéo annotée")
            video_button = gr.Button("Analyser la vidéo", variant="primary")
            # Appel inchangé
            video_button.click(fn=predict_video, inputs=video_input, outputs=video_output)

        with gr.Column():
            gr.Markdown("### Détection sur **image**")
            image_input = gr.Image(type="pil", label="Uploader une image")
            image_output = gr.Image(label="Image annotée")
            image_button = gr.Button("Analyser l'image", variant="primary")
            # Appel inchangé
            image_button.click(fn=predict_image, inputs=image_input, outputs=image_output)

    # Pied de page discret (crédit & disclaimer)
    gr.HTML("""
    <div style="margin-top:14px;padding:10px 12px;border:1px solid #e2e8f0;border-radius:12px;background:#f8fbff;">
      <div style="font-size:13px;color:#2d3748;">
        ⚠️ <b>Disclaimer :</b> Stroke-IA est une démonstration technique. Les résultats ne constituent pas un avis médical
        et doivent être validés par un professionnel de santé.
      </div>
      <div style="margin-top:6px;font-size:12px;color:#4a5568;">
        © {year} — Badsi Djilali. Tous droits réservés.
      </div>
    </div>
    """.format(year=datetime.now().year))

demo.launch(share=True)

