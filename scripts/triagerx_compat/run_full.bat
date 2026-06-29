@echo off
cd /d C:\Users\OMEN\Documents\Programacion\Python\triagerX
set PYTHONPATH=C:\Users\OMEN\Documents\Programacion\Python\triagerX
.venv\Scripts\python.exe omega_split\scripts\train_triagerx_omega.py ^
  --triagerx-root C:\Users\OMEN\Documents\Programacion\Python\triagerX ^
  --config training\training_config\openj9\developer\triagerx.yaml ^
  --train-csv omega_split\openj9_train_50.csv ^
  --test-csv omega_split\openj9_test_50.csv ^
  --val-csv omega_split\openj9_val_50.csv ^
  --out-dir omega_split\runs ^
  --threshold 1 --batch-size 8 --grad-checkpoint > omega_split\runs\train_full.log 2>&1
echo DONE_EXIT_%ERRORLEVEL%>> omega_split\runs\train_full.log
