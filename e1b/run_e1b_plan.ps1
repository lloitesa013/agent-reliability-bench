# E1-B plan runner (E1B_PREREG.md). Executes, in order:
#   1) strict determinism probe : det=1 + E1B_STRICT=1  (C7-b; a raise IS the localization)
#   2) det=1 pair               : runs a, b  (warn_only)
#   3) det=0 pair               : runs a, b
# Route default 11755. CARLA DEFAULT PHYSICS -- E1_L2 is deliberately never set (C6: the
# 0.0125x4 spec changes the integrator from 5 substeps to 4; it is a perturbation, not a lock).
#
# C3 corrections applied vs the E1 runners: per-run watchdog 2400s (was 1200s, which fired on the
# slow failing branch and produced non-random dropout); output directories are NEVER deleted before
# a retry (that destroyed the evidence of what happened); every attempt is recorded with its
# termination reason in an attempt ledger, so censoring is visible instead of silent.
param([string]$Route = "11755", [int]$Pairs = 1)
$ErrorActionPreference = "Continue"
Start-Transcript -Path "C:\lead\outputs\e1b_master.log" -Append
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
Remove-Item Env:E1_L2 -ErrorAction SilentlyContinue      # never set: default physics
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$WALL = 2400
$root = Join-Path $Lead "outputs\e1b"
New-Item -ItemType Directory -Force -Path $root | Out-Null
$ledger = Join-Path $root "attempts.csv"
if (-not (Test-Path $ledger)) {
  "route,cell,runid,attempt,started,ended,det,strict,termination,status,score,collisions,tick_rows,gpu_mb" |
    Out-File -Encoding utf8 $ledger
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
  if (-not (Stop-CarlaHard)) { return $false }
  schtasks /run /tn CarlaServerNewLog | Out-Null
  for ($i = 1; $i -le 120; $i++) {
    if (Test-Port2000) { Start-Sleep -Seconds 20; return (Test-Port2000) }
    Start-Sleep -Seconds 1
  }
  return $false
}

function Invoke-E1BRun([string]$Cell, [string]$RunId, [int]$Det, [int]$Strict) {
  for ($attempt = 1; $attempt -le 2; $attempt++) {
    $tag = "{0}_{1}_{2}_t{3}" -f $Route, $Cell, $RunId, $attempt
    $out = Join-Path $root $tag
    $cp = Join-Path $out "checkpoint_endpoint.json"
    if (Test-Path $cp) {
      try {
        $j0 = Get-Content -Raw $cp | ConvertFrom-Json
        if (-not [string]::IsNullOrWhiteSpace($j0._checkpoint.global_record.status)) {
          Write-Output ("SKIP valid {0}" -f $tag); return
        }
      } catch {}
    }
    $ticks = Join-Path $root ("{0}_ticks.csv" -f $tag)
    if ($Det -eq 1) { $env:E1_DETERMINISM = "1"; $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8" }
    else { Remove-Item Env:E1_DETERMINISM -ErrorAction SilentlyContinue }
    if ($Strict -eq 1) { $env:E1B_STRICT = "1" } else { Remove-Item Env:E1B_STRICT -ErrorAction SilentlyContinue }
    $env:E1B_TICKLOG = $ticks

    $started = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Write-Output ("=== E1B {0} cell={1} run={2} attempt={3} det={4} strict={5} {6} ===" -f $Route, $Cell, $RunId, $attempt, $Det, $Strict, $started)
    if (-not (Start-CarlaFresh)) {
      ("{0},{1},{2},{3},{4},{5},{6},{7},CARLA_UNSTABLE,,,,," -f $Route, $Cell, $RunId, $attempt, $started, (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $Det, $Strict) | Out-File -Append -Encoding utf8 $ledger
      continue
    }
    $gpu = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join ""
    New-Item -ItemType Directory -Force -Path $out | Out-Null    # never deleted: evidence preserved
    Set-Location $Lead
    $log = Join-Path $root ("{0}.log" -f $tag)
    $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/tfv6_resnet34",
      "--routes", "data/benchmark_routes/bench2drive/$Route.xml", "--bench2drive",
      "--port", "2000", "--traffic-manager-seed", "0",
      "--timeout", "900", "--output-dir", $out)
    $p = Start-Process -FilePath $Python -ArgumentList $rargs -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $deadline = (Get-Date).AddSeconds($WALL)
    while ((-not $p.HasExited) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 10 }
    $term = "EXITED"
    if (-not $p.HasExited) {
      $term = "WATCHDOG_2400"
      try { $p.Kill() } catch {}
      Stop-CarlaHard | Out-Null
    }
    $ended = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    $rows = 0
    if (Test-Path $ticks) { $rows = (Get-Content $ticks | Measure-Object -Line).Lines }
    $status = ""; $score = ""; $col = ""
    if (Test-Path $cp) {
      try {
        $j = Get-Content -Raw $cp | ConvertFrom-Json
        $gr = $j._checkpoint.global_record
        $status = $gr.status
        $score = $gr.scores_mean.score_composed
        $col = [double]$gr.infractions.collisions_vehicle + [double]$gr.infractions.collisions_pedestrian + [double]$gr.infractions.collisions_layout
      } catch { $status = "PARSE_ERR" }
    }
    # C7-b: a strict-mode determinism error is a RESULT, not a failure -- capture it verbatim
    $strictErr = ""
    if (Test-Path "$log.err") {
      $m = Select-String -Path "$log.err" -Pattern "does not have a deterministic implementation|deterministic algorithm" -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($m) { $strictErr = "DETERMINISM_RAISE"; Write-Output ("  C7-b RAISE: " + $m.Line.Trim()) }
    }
    if ($strictErr) { $term = $strictErr }
    ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}" -f $Route, $Cell, $RunId, $attempt, $started, $ended, $Det, $Strict, $term, $status, $score, $col, $rows, $gpu) |
      Out-File -Append -Encoding utf8 $ledger
    Write-Output ("  term={0} status={1} score={2} col={3} ticks={4}" -f $term, $status, $score, $col, $rows)
    if ($status -and $status -ne "PARSE_ERR") { return }
  }
}

Invoke-E1BRun -Cell "strict" -RunId "s1" -Det 1 -Strict 1
for ($i = 1; $i -le $Pairs; $i++) {
  Invoke-E1BRun -Cell "det1" -RunId ("a{0}" -f $i) -Det 1 -Strict 0
  Invoke-E1BRun -Cell "det1" -RunId ("b{0}" -f $i) -Det 1 -Strict 0
  Invoke-E1BRun -Cell "det0" -RunId ("a{0}" -f $i) -Det 0 -Strict 0
  Invoke-E1BRun -Cell "det0" -RunId ("b{0}" -f $i) -Det 0 -Strict 0
}
Stop-CarlaHard | Out-Null
Write-Output "E1B PLAN ALL DONE"
Stop-Transcript
