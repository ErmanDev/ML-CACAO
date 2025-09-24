import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import pickle

# --- Load datasets ---
train_df = pd.read_csv("features_training.csv")
val_df   = pd.read_csv("features_validation.csv")
calib_df = pd.read_csv("features_calibration.csv")

# Split features and labels
X_train, y_train = train_df.drop("Label", axis=1), train_df["Label"]
X_val, y_val     = val_df.drop("Label", axis=1), val_df["Label"]
X_calib, y_calib = calib_df.drop("Label", axis=1), calib_df["Label"]

# --- Train base model ---
print("[INFO] Training Random Forest...")
base_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
base_model.fit(X_train, y_train)


y_pred = base_model.predict(X_val)
print("\n[INFO] Validation Results (before calibration):")
print(classification_report(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))


print("\n[INFO] Calibrating model with calibration set...")
calibrated_model = CalibratedClassifierCV(estimator=base_model, cv="prefit")
calibrated_model.fit(X_calib, y_calib)


y_pred_calib = calibrated_model.predict(X_val)
print("\n[INFO] Validation Results (after calibration):")
print(classification_report(y_val, y_pred_calib))
print("Accuracy:", accuracy_score(y_val, y_pred_calib))


with open("models/cacao_model.pkl", "wb") as f:
    pickle.dump(calibrated_model, f)

print("\n[INFO] Final calibrated model saved to models/cacao_model.pkl")