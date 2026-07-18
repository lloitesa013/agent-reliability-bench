# S4-(3) blind-arm trainer, parameterized. Example:
#   train_s4b.ps1 -Name S4-B1 -Epochs 5 -CarlaRoot data/vsi_broad_t -Ratio none -Freeze 1
param(
  [string]$Name,
  [int]$Epochs,
  [string]$CarlaRoot,
  [string]$Ratio,      # "none" | "0.4" | "0.6"
  [int]$Freeze         # 0 | 1
)
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$env:LEAD_PROJECT_ROOT = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:PYTHONPATH = "C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla;C:\lead"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$env:PATH"
$env:OMP_NUM_THREADS = "8"; $env:WANDB_MODE = "offline"; $env:USE_LIBUV = "0"
$env:SCRATCH = "C:\lead\tmp_cache"; $env:USER = "tulpa"; $env:USERNAME = "tulpa"
if ($Ratio -eq "none") {
  Remove-Item Env:VSI_TWO_BUCKET -ErrorAction SilentlyContinue
} else {
  $env:VSI_TWO_BUCKET = "1"; $env:VSI_FIX_SCENARIOS = "EnterActorFlow"
  $env:VSI_FIX_ROUTES = "_11755_"; $env:VSI_FIX_RATIO = $Ratio
  $env:VSI_RET_RATIO = ([string](1.0 - [double]$Ratio))
}
$fz = if ($Freeze -eq 1) { " freeze_backbone=true" } else { "" }
$logdir = "outputs/local_training/vsi_" + $Name.ToLower().Replace("-", "")
$env:LEAD_TRAINING_CONFIG = "logdir=$logdir load_file=outputs/checkpoints/tfv6_resnet34/model_0030_0.pth carla_root=$CarlaRoot use_planning_decoder=true epochs=$Epochs assigned_cpu_cores=1 force_rebuild_bucket=true$fz"
Set-Location C:\lead
& "$EnvDir\python.exe" lead\training\train.py
Write-Output ("exit=" + $LASTEXITCODE)
