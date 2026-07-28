# E1 generic level runner (L1+). Same validity/resume/watchdog discipline as run_e1_l0d.ps1.
#   -Tag       output namespace, e.g. l1 / l1seedvary / l2 / l3   -> outputs\e1_<Tag>\
#   -SeedMode  fixed (default, uses -Seed) | vary (TM seed = run index k)
#   -Seed      TM seed when SeedMode=fixed (L0's effective value was 0)
#   -Routes    default 11755,18252,3436     -Runs default 12
#   -Determinism  1 => sets E1_DETERMINISM=1 for the L3 rung (agent-side torch locks)
# Machine state (GPU memory in use) is logged per run -- R52 flagged session/machine drift as a
# confound, so from L1 on it is measured rather than assumed.
param(
  [string]$Tag = "l1",
  [string]$SeedMode = "fixed",
  [int]$Seed = 0,
  [string[]]$Routes = @("11755", "18252", "3436"),
  [int]$Runs = 12,
  [int]$Determinism = 0
)
$ErrorActionPreference = "Continue"
Start-Transcript -Path ("C:\lead\outputs\e1_{0}_master.log" -f $Tag) -Append
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
if ($Determinism -eq 1) { $env:E1_DETERMINISM = "1"; $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8" }
else { Remove-Item Env:E1_DETERMINISM -ErrorAction SilentlyContinue }
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$perRunTimeoutSec = 1200
$outRoot = Join-Path $Lead ("outputs\e1_{0}" -f $Tag)
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
$machCsv = Join-Path $outRoot "machine.csv"
if (-not (Test-Path $machCsv)) { "route,run,seed,ts,gpu_mem_used_mb,gpu_mem_total_mb" | Out-File -Encoding utf8 $machCsv }

function Test-ValidCheckpoint($cpPath) {
  if (-not (Test-Path $cpPath)) { return $false }
  try {
    $j = Get-Content -Raw $cpPath | ConvertFrom-Json
    return (-not [string]::IsNullOrWhiteSpace($j._checkpoint.global_record.status))
  } catch { return $false }
}
function Test-Port2000 { return (Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded }
function Stop-CarlaHard {
  Get-Process CarlaUE4, CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  schtasks /end /tn CarlaServerNewLog | Out-Null
  Start-Sleep -Seconds 3
  Get-Process CarlaUE4, CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  for ($i = 1; $i -le 40; $i++) { if (-not (Test-Port2000)) { return $true }; Start-Sleep -Seconds 1 }
  return $false
}
function Start-CarlaFresh {
  if (-not (Stop-CarlaHard)) { Write-Output "  port 2000 still held -> abort try"; return $false }
  schtasks /run /tn CarlaServerNewLog | Out-Null
  for ($i = 1; $i -le 120; $i++) {
    if (Test-Port2000) { Start-Sleep -Seconds 20; return (Test-Port2000) }
    Start-Sleep -Seconds 1
  }
  return $false
}

for ($k = 1; $k -le $Runs; $k++) {
  foreach ($RouteId in $Routes) {
    $tmSeed = if ($SeedMode -eq "vary") { $k } else { $Seed }
    $out = Join-Path $outRoot ("{0}_run{1}" -f $RouteId, $k)
    $cp = Join-Path $out "checkpoint_endpoint.json"
    if (Test-ValidCheckpoint $cp) { Write-Output ("=== SKIP valid: {0} {1} run {2}/{3} ===" -f $Tag, $RouteId, $k, $Runs); continue }
    $done = $false
    for ($try = 1; $try -le 2 -and -not $done; $try++) {
      Write-Output ("=== E1-{0} route {1} run {2}/{3} seed={4} det={5} (try {6}) {7} ===" -f $Tag, $RouteId, $k, $Runs, $tmSeed, $Determinism, $try, (Get-Date -Format "MM-dd HH:mm:ss"))
      if (-not (Start-CarlaFresh)) { Write-Output "  CARLA not stable -> next try"; continue }
      $gpu = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -join ","
      ("{0},{1},{2},{3},{4}" -f $RouteId, $k, $tmSeed, (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $gpu) | Out-File -Append -Encoding utf8 $machCsv
      if (Test-Path $out) { Remove-Item -Recurse -Force $out }
      New-Item -ItemType Directory -Force -Path $out | Out-Null
      Set-Location $Lead
      $log = Join-Path $outRoot ("{0}_run{1}_t{2}.log" -f $RouteId, $k, $try)
      $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/tfv6_resnet34",
        "--routes", "data/benchmark_routes/bench2drive/$RouteId.xml", "--bench2drive",
        "--port", "2000", "--traffic-manager-seed", "$tmSeed",
        "--timeout", "900", "--output-dir", $out)
      $p = Start-Process -FilePath $Python -ArgumentList $rargs -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
      $deadline = (Get-Date).AddSeconds($perRunTimeoutSec)
      while ((-not $p.HasExited) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 10 }
      if (-not $p.HasExited) {
        Write-Output ("  TIMEOUT {0}s -> kill" -f $perRunTimeoutSec)
        try { $p.Kill() } catch {}
        Stop-CarlaHard | Out-Null
        continue
      }
      if (Test-ValidCheckpoint $cp) {
        $j = Get-Content -Raw $cp | ConvertFrom-Json
        $gr = $j._checkpoint.global_record
        $col = $gr.infractions.collisions_vehicle + $gr.infractions.collisions_pedestrian + $gr.infractions.collisions_layout
        Write-Output ("  RESULT status={0} score={1} collisions={2}" -f $gr.status, $gr.scores_mean.score_composed, $col)
        $done = $true
      } else { Write-Output "  RESULT invalid/no checkpoint" }
    }
  }
}
Stop-CarlaHard | Out-Null
Write-Output ("E1-{0} ALL DONE" -f $Tag)
Stop-Transcript
