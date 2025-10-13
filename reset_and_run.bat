@echo off
echo  Resetting Cacao Bean Classifier...

REM Step 1: Remove old feature CSVs
echo  Removing feature CSVs...
del /Q features_training.csv features_validation.csv features_calibration.csv

REM Step 2: Remove old model
echo  Removing trained model...
del /Q models\cacao_model.pkl

REM Step 3: Clear uploaded images
echo  Clearing uploaded images...
del /Q static\uploads\*

REM Step 4: Rebuild features
echo  Rebuilding features...
python build_features.py

REM Step 5: Retrain and calibrate model
echo Retraining and calibrating model...
python train_model.py

REM Step 6: Launch the web app
echo Launching Flask app...
python app.py

echo Fresh start complete. Visit http://localhost:5000 to test your classifier!
pause