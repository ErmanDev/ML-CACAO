import cv2
import numpy as np
from scipy.stats import skew
from pathlib import Path
import csv

# --- HSV Feature Extraction ---
def extract_hsv_features(img_path):
    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    def stats(channel):
        return [
            np.mean(channel),
            np.std(channel),
            np.var(channel),
            skew(channel.flatten())
        ]

    return stats(h) + stats(s) + stats(v)

# --- Process a Dataset Split ---
def process_split(split_name, base_dir="data"):
    split_dir = Path(base_dir) / split_name
    output_file = f"features_{split_name.lower()}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            'Hue_mean', 'Hue_std', 'Hue_var', 'Hue_skew',
            'Sat_mean', 'Sat_std', 'Sat_var', 'Sat_skew',
            'Val_mean', 'Val_std', 'Val_var', 'Val_skew',
            'Label'
        ]
        writer.writerow(header)

        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                label = class_dir.name.upper()
                for img_path in class_dir.glob("*.jpg"):
                    try:
                        features = extract_hsv_features(img_path)
                        writer.writerow(features + [label])
                    except Exception as e:
                        print(f"[WARN] Skipped {img_path.name}: {e}")

    print(f"[INFO] Saved {split_name} features to {output_file}")

# --- Main ---
if __name__ == "__main__":
    for split in ["Training", "Validation", "Calibration"]:
        process_split(split)