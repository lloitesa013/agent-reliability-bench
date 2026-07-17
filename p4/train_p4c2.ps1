# P4-c2: transfer-known-good (A2 mixture) to target 28330. fix 60% / retention 40%, 5 epochs, full FT.
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$env:LEAD_PROJECT_ROOT = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:PYTHONPATH = "C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla;C:\lead"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Libraryin;$env:PATH"
$env:OMP_NUM_THREADS = "8"; $env:WANDB_MODE = "offline"; $env:USE_LIBUV = "0"
$env:VSI_TWO_BUCKET = "1"; $env:VSI_FIX_SCENARIOS = "SignalizedJunctionLeftTurnEnterFlow"
$env:VSI_FIX_ROUTES = "_28330_"; $env:VSI_FIX_RATIO = "0.6"; $env:VSI_RET_RATIO = "0.4"
$env:LEAD_TRAINING_CONFIG = "logdir=outputs/local_training/vsi_p4c2 load_file=outputs/checkpoints/tfv6_resnet34/model_0030_0.pth carla_root=data/carla_leaderboard2_vsi use_planning_decoder=true epochs=5 assigned_cpu_cores=1 force_rebuild_bucket=true"
Set-Location C:\lead
& "$EnvDir\python.exe" lead	raining	rain.py
Write-Output ("exit=" + $LASTEXITCODE)
