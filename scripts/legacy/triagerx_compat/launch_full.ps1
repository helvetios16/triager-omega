$root = 'C:\Users\OMEN\Documents\Programacion\Python\triagerX'
$env:PYTHONPATH = $root
$py = "$root\.venv\Scripts\python.exe"
$pyargs = @(
  "$root\omega_split\scripts\train_triagerx_omega.py",
  '--triagerx-root', $root,
  '--config', "$root\training\training_config\openj9\developer\triagerx.yaml",
  '--train-csv', "$root\omega_split\openj9_train_50.csv",
  '--test-csv', "$root\omega_split\openj9_test_50.csv",
  '--val-csv', "$root\omega_split\openj9_val_50.csv",
  '--out-dir', "$root\omega_split\runs",
  '--threshold', '1', '--batch-size', '8', '--grad-checkpoint'
)
$out = "$root\omega_split\runs\train_full.out.log"
$err = "$root\omega_split\runs\train_full.err.log"
$p = Start-Process -FilePath $py -ArgumentList $pyargs -WorkingDirectory $root `
     -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
Write-Output ("STARTED_PID " + $p.Id)
