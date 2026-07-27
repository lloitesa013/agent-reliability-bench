# E1-L0d: L0 baseline. Supersedes run_e1_l0c.ps1 (l0c kept for history; do not run).
#
# Changes vs l0c (ops robustness only -- NO sealed quantity touched: same 3 routes, same N=12,
# same evaluator, same decision lines in E1_PREREG.md e46c94d):
#   (1) KILL BOTH process names. l0c killed only "CarlaUE4"; the real server process is
#       "CarlaUE4-Win64-Shipping" and escaped every kill -- the same zombie that produced a
#       10-run invalid streak in the S4 chain. A surviving zombie holds port 2000 LISTENING,
#       so l0c's port probe succeeded instantly against a STALE server (trap #1's root cause).
#   (2) WAIT FOR PORT 2000 TO GO FREE after /end before /run, so a bind can never be attributed
#       to the previous server. Port-free is now a precondition, not an assumption.
#   (3) ROUND-ROBIN route order (k=1: 11755,18252,3436; k=2: ...). Two reasons: this campaign
#       keeps getting interrupted, and balanced partial data beats one complete route; and it
#       decorrelates route from wall-clock/thermal drift, which matters for a study ABOUT
#       stochasticity. Every route still gets its sealed 12 runs.
# Unchanged: valid-checkpoint resume, 20s post-bind stabilization + re-test, 1 retry per run,
# 1200s watchdog via $p.HasExited polling (WaitForExit(ms) ignores its timeout -- known bug).
$ErrorActionPreference = "Continue"
Start-Transcript -Path "C:\lead\outputs\e1_l0d_master.log" -Append
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$routes = @("11755", "18252", "3436")
$N = 12
$perRunTimeoutSec = 1200

function Test-ValidCheckpoint($cpPath) {
  if (-not (Test-Path $cpPath)) { return $false }
  try {
    $j = Get-Content -Raw $cpPath | ConvertFrom-Json
    $s = $j._checkpoint.global_record.status
    return (-not [string]::IsNullOrWhiteSpace($s))
  } catch { return $false }
}

function Test-Port2000 {
  return (Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded
}

function Stop-CarlaHard {
  Get-Process CarlaUE4, CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
  schtasks /end /tn CarlaServerNewLog | Out-Null
  Start-Sleep -Seconds 3
  Get-Process CarlaUE4, CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
  # port must actually go FREE -- otherwise the next bind is the old server
  for ($i = 1; $i -le 40; $i++) {
    if (-not (Test-Port2000)) { return $true }
    Start-Sleep -Seconds 1
  }
  return $false
}

function Start-CarlaFresh {
  if (-not (Stop-CarlaHard)) { Write-Output "  port 2000 still held after kill -> abort this try"; return $false }
  schtasks /run /tn CarlaServerNewLog | Out-Null
  for ($i = 1; $i -le 120; $i++) {
    if (Test-Port2000) {
      Start-Sleep -Seconds 20
      if (Test-Port2000) { return $true }     # survived town load
      return $false                            # bound then died (VRAM pressure pattern)
    }
    Start-Sleep -Seconds 1
  }
  return $false
}

for ($k = 1; $k -le $N; $k++) {
  foreach ($RouteId in $routes) {
    $out = Join-Path $Lead ("outputs\e1_l0\{0}_run{1}" -f $RouteId, $k)
    $cp  = Join-Path $out "checkpoint_endpoint.json"
    if (Test-ValidCheckpoint $cp) { Write-Output ("=== SKIP valid: route {0} run {1}/{2} ===" -f $RouteId, $k, $N); continue }
    $done = $false
    for ($try = 1; $try -le 2 -and -not $done; $try++) {
      Write-Output ("=== E1-L0d route {0} run {1}/{2} (try {3}) {4} ===" -f $RouteId, $k, $N, $try, (Get-Date -Format "MM-dd HH:mm:ss"))
      if (-not (Start-CarlaFresh)) { Write-Output "  CARLA not stable -> next try"; continue }
      if (Test-Path $out) { Remove-Item -Recurse -Force $out }
      New-Item -ItemType Directory -Force -Path $out | Out-Null
      Set-Location $Lead
      $log = Join-Path $Lead ("outputs\e1_l0\{0}_run{1}_t{2}.log" -f $RouteId, $k, $try)
      $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/tfv6_resnet34",
        "--routes", "data/benchmark_routes/bench2drive/$RouteId.xml", "--bench2drive",
        "--port", "2000", "--timeout", "900", "--output-dir", $out)
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
        Write-Output ("  RESULT status={0} score={1} collisions={2} route_dev={3}" -f $gr.status, $gr.scores_mean.score_composed, $col, $gr.infractions.route_dev)
        $done = $true
      } else { Write-Output "  RESULT invalid/no checkpoint" }
    }
  }
}
Stop-CarlaHard | Out-Null
Write-Output "E1-L0d ALL DONE"
Stop-Transcript
