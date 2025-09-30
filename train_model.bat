@echo off

@REM SET model_name=Ensemble
@REM SET mode=ensemble
@REM SET ensemble_size=1

python train.py --MODEL_PATH trained/ensemble/Ensemble --MODEL_NAME 1 --DOWN_DROP 0,0,0,0 --UP_DROP 0,0,0,0
