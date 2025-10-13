@echo off
echo Starting Cacao Bean Classifier Setup...

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Extracting features from images...
python build_features.py

echo Training and calibrating model...
python train_model.py

echo Launching Flask app...
python app.py

echo Setup complete. Visit http://localhost:5000 to test your classifier!