# Embodied failure-rate verifier WITH A HARD WALL-CLOCK WATCHDOG (fixes the R21 wedge: LEAD --timeout is
# only an internal scenario timeout and does NOT kill a hung CARLA-blocked python). Each run is a child
# process; if it exceeds WALL_LIMIT_MS wall-clock, we force-kill python + CarlaUE4 and move on. Never wedges.
$ErrorActionPreference = "Continue"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$WALL_LIMIT_MS = 1200000   # 20 min hard wall-clock per run (route runs ~2-4 min normally)
$plan = @(@("11755", 6))   # edit per experiment
$csv = Join-Path $Lead "outputs\failure_rate_wd.csv"
"route,run,status,score,collisions,failed,watchdog" | Out-File -Encoding utf8 $csv
$results = @()

foreach ($item in $plan) {
  $RouteId = $item[0]; $N = [int]$item[1]
  for ($k = 1; $k -le $N; $k++) {
    Write-Output ("=== route {0} run {1}/{2} ({3}) ===" -f $RouteId, $k, $N, (Get-Date -Format HH:mm:ss))
    Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    schtasks /end /tn CarlaServerNewLog | Out-Null
    Start-Sleep -Seconds 5
    schtasks /run /tn CarlaServerNewLog | Out-Null
    $up = $false
    for ($i = 1; $i -le 120; $i++) {
      if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
      Start-Sleep -Seconds 1
    }
    if (!$up) { Write-Output "  CARLA port failed, skip"; "$RouteId,$k,PORT_FAIL,,,NA,0" | Out-File -Append -Encoding utf8 $csv; continue }
    $out = Join-Path $Lead ("outputs\local_evaluation_win\{0}_wd{1}" -f $RouteId, $k)
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $Lead ("outputs\wd_{0}_run{1}.log" -f $RouteId, $k)
    $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/tfv6_resnet34",
      "--routes", "data/benchmark_routes/bench2drive/$RouteId.xml", "--bench2drive",
      "--port", "2000", "--timeout", "900", "--output-dir", $out)
    # HARD WATCHDOG: run as a child, wait at most WALL_LIMIT_MS, force-kill on expiry
    $wd = 0
    $proc = Start-Process -FilePath $Python -ArgumentList $rargs -WorkingDirectory $Lead -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if (-not $proc.WaitForExit($WALL_LIMIT_MS)) {
      $wd = 1
      Write-Output ("  WATCHDOG: run exceeded {0}min - force-killing python+CARLA" -f ($WALL_LIMIT_MS / 60000))
      try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
      Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
      Start-Sleep -Seconds 3
    }
    $cp = Join-Path $out "checkpoint_endpoint.json"
    if (Test-Path $cp) {
      $j = Get-Content -Raw $cp | ConvertFrom-Json
      $gr = $j._checkpoint.global_record
      $col = [double]$gr.infractions.collisions_vehicle + [double]$gr.infractions.collisions_pedestrian + [double]$gr.infractions.collisions_layout
      $score = [double]$gr.scores_mean.score_composed
      $failed = if (($score -lt 100) -or ($col -gt 0)) { 1 } else { 0 }
      Write-Output ("  RESULT status={0} score={1} collisions={2} failed={3} watchdog={4}" -f $gr.status, $score, $col, $failed, $wd)
      "$RouteId,$k,$($gr.status),$score,$col,$failed,$wd" | Out-File -Append -Encoding utf8 $csv
      $results += [pscustomobject]@{ route = $RouteId; failed = $failed }
    } else {
      Write-Output ("  RESULT no checkpoint (watchdog={0})" -f $wd)
      "$RouteId,$k,NO_CHECKPOINT,,,NA,$wd" | Out-File -Append -Encoding utf8 $csv
    }
  }
}
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output "`n=== FAILURE-RATE (hard-watchdog runner) ==="
foreach ($item in $plan) {
  $RouteId = $item[0]
  $runs = $results | Where-Object { $_.route -eq $RouteId }
  $nn = ($runs | Measure-Object).Count
  $ff = ($runs | Where-Object { $_.failed -eq 1 } | Measure-Object).Count
  if ($nn -gt 0) { Write-Output ("route {0}: FAILED {1}/{2} = {3:P0}" -f $RouteId, $ff, $nn, ($ff / $nn)) }
}
Write-Output "ALL DONE (watchdog runner - self-cleans CARLA at end)"
