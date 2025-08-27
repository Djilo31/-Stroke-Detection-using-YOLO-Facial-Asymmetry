# Stroke-IA 🧠
Prototype d’IA pour la détection d’AVC via webcam et images.  
⚠️ Cet outil est un prototype de démonstration – il ne remplace pas un avis médical.  

**Auteur** : Badsi Djilali – Ingénieur IA Deep Learning  


🧠 Stroke-IA — AI Prototype for Stroke Sign Detection
Built a YOLOv8-based model with a Gradio interface for analyzing images, videos, and webcam streams to detect early stroke indicators.
# Stroke Detect YOLO

Détection précoce d'AVC basé sur l'asymétrie faciale (yeux et bouche) grâce à YOLOv8.

## Contenu
- `best.pt` : Modèle YOLOv8 entraîné
- `app.py` : Script de détection
- `requirements.txt` : Dépendances Python

## Utilisation
```bash
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("image.jpg", conf=0.5)
results.show()

⚠️ Research prototype — not for medical use.
