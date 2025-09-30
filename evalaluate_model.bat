@echo off

SET model_name=Ensemble
SET mode=ensemble
SET ensemble_size=1


python generate_predictions.py --MODE %mode% --MODEL_PATH "trained\%mode%/%model_name%"
python evaluate.py --MODE %mode% --SAMPLES %ensemble_size% --MODEL_NAME %model_name%