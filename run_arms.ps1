# Embodied FIX PROBE: run route 11755 under throttle-scale interventions, measure failure RATE per arm
# (the statistical verifier applied to a real control-layer "fix"). LEAD_THROTTLE_SCALE drives the injected
# control in sensor_agent.py (1.0 = unmodified tfv6; <1 conservative = fix candidate; >1 aggressive = regression
# candidate). Hard 10-min wall-clock watchdog per run. Flat $scales list avoids PowerShell array flattening.
$ErrorActionPreference = "Continue"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$WALL_LIMIT_MS = 600000        # 10 min hard kill (normal run ~2-4 min)
$Route = "11755"
$scales = @("1.0", "0.7", "1.3")
$N = 6
$csv = Join-Path $Lead "outputs\arms_$Route.csv"
"scale,run,status,score,collisions,failed,watchdog" | Out-File -Encoding utf8 $csv
$results = @()

foreach ($scale in $scales) {
  $env:LEAD_THROTTLE_SCALE = $scale
  for ($k = 1; $k -le $N; $k++) {
    Write-Output ("=== scale {0} route {1} run {2}/{3} ({4}) ===" -f $scale, $Route, $k, $N, (Get-Date -Format HH:mm:ss))
    Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    schtasks /end /tn CarlaServerNewLog | Out-Null
    Start-Sleep -Seconds 5
    schtasks /run /tn CarlaServerNewLog | Out-Null
    $up = $false
    for ($i = 1; $i -le 120; $i++) {
      if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
      Start-Sleep -Seconds 1
    }
    if (!$up) { Write-Output "  CARLA port failed, skip"; "$scale,$k,PORT_FAIL,,,NA,0" | Out-File -Append -Encoding utf8 $csv; continue }
    $tag = ($scale -replace '\.', '')
    $out = Join-Path $Lead ("outputs\local_evaluation_win\{0}_s{1}_r{2}" -f $Route, $tag, $k)
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $Lead ("outputs\arm_{0}_s{1}_r{2}.log" -f $Route, $tag, $k)
    $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/tfv6_resnet34",
      "--routes", "data/benchmark_routes/bench2drive/$Route.xml", "--bench2drive",
      "--port", "2000", "--timeout", "900", "--output-dir", $out)
    $wd = 0
    $proc = Start-Process -FilePath $Python -ArgumentList $rargs -WorkingDirectory $Lead -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if (-not $proc.WaitForExit($WALL_LIMIT_MS)) {
      $wd = 1; Write-Output "  WATCHDOG kill (>10min)"
      try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
      Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
      Start-Sleep -Seconds 3
    }
    $cp = Join-Path $out "checkpoint_endpoint.json"
    if ((-not $wd) -and (Test-Path $cp)) {
      $j = Get-Content -Raw $cp | ConvertFrom-Json
      $gr = $j._checkpoint.global_record
      $col = [double]$gr.infractions.collisions_vehicle + [double]$gr.infractions.collisions_pedestrian + [double]$gr.infractions.collisions_layout
      $score = [double]$gr.scores_mean.score_composed
      $failed = if (($score -lt 100) -or ($col -gt 0)) { 1 } else { 0 }
      Write-Output ("  RESULT scale={0} status={1} score={2} collisions={3} failed={4}" -f $scale, $gr.status, $score, $col, $failed)
      "$scale,$k,$($gr.status),$score,$col,$failed,0" | Out-File -Append -Encoding utf8 $csv
      $results += [pscustomobject]@{ scale = $scale; failed = $failed }
    } else {
      Write-Output ("  RESULT invalid (watchdog={0}) - excluded from rate" -f $wd)
      "$scale,$k,INVALID,,,NA,$wd" | Out-File -Append -Encoding utf8 $csv
    }
  }
}
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output "`n=== FAILURE RATE BY THROTTLE SCALE (route $Route, valid runs only) ==="
foreach ($scale in $scales) {
  $runs = $results | Where-Object { $_.scale -eq $scale }
  $nn = ($runs | Measure-Object).Count
  $ff = ($runs | Where-Object { $_.failed -eq 1 } | Measure-Object).Count
  if ($nn -gt 0) { Write-Output ("scale {0}: FAILED {1}/{2} = {3:P0}" -f $scale, $ff, $nn, ($ff / $nn)) }
  else { Write-Output ("scale {0}: no valid runs" -f $scale) }
}
Write-Output "ALL DONE"
