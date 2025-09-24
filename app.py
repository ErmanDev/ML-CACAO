from flask import Flask, render_template, request
import os
import cv2
import pickle
import numpy as np
from utils import auto_crop_contours, extract_hsv_features




# --- Flask setup ---
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Load trained model ---
with open("models/cacao_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load image
    img = cv2.imread(filepath)
    
    # Convert to HSV and split channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

# Save each channel as grayscale preview
    cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "hue.jpg"), h)
    cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "saturation.jpg"), s)
    cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "value.jpg"), v)  

    # Auto-crop beans (adjust rows/cols if needed)
    beans = auto_crop_contours(img)
    if len(beans) == 0:
        beans = [img]
    results = []
    bean_images = []

    for i, bean in enumerate(beans):
        features = extract_hsv_features(bean)
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        confidence = np.max(proba) * 100

    # Save cropped bean image
        bean_filename = f"bean_{i+1}.jpg"
        bean_path = os.path.join(app.config["UPLOAD_FOLDER"], bean_filename)
        cv2.imwrite(bean_path, bean)
        bean_images.append(bean_filename)

        results.append({
            "text": f"Bean {i+1}: {prediction} ({confidence:.2f}% confidence)",
            "image": bean_filename
        })

    return render_template("index.html", results=results, image=file.filename)

    

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)