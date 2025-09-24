# 🍫 Cacao Bean Fermentation Classifier

A machine learning web app that classifies cacao beans based on their fermentation quality using HSV color features. Built with Python, Flask, OpenCV, and scikit-learn.

---

## 🚀 Features

- Upload cacao bean images for classification
- Dynamic bean cropping using contour detection
- HSV channel previews (Hue, Saturation, Value)
- Confidence scores for each prediction
- Optional calibration for probability adjustment
- Clean Bootstrap-based frontend

---

##  Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt

python build_features.py

python train_model.py

python app.py

---

## Reset Workflow (Optional)
To start fresh:
- Delete old CSVs in root folder
- Delete models/cacao_model.pkl
- Clear static/uploads/
- Re-run steps 2–4


