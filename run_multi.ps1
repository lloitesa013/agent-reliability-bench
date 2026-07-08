# Embodied STATISTICAL failure-rate verifier: re-run flaky routes N times each (fresh CARLA per run) to
# turn a single-run PASS/FAIL (meaningless - driving failures are flaky) into a failure RATE. This is the
# embodied analog of the text hidden-test: verify by rate over N runs, not one run.
$ErrorActionPreference = "Continue"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

# 11755 = EnterActorFlow (documented flaky: collision->PASS on re-run); 3436 = HazardSideLane (3 baseline collisions)
$plan = @(@("11755", 6), @("3436", 4))
$csv = Join-Path $Lead "outputs\failure_rate.csv"
"route,run,status,score,collisions,route_dev,failed" | Out-File -Encoding utf8 $csv
$results = @()

foreach ($item in $plan) {
  $RouteId = $item[0]; $N = [int]$item[1]
  for ($k = 1; $k -le $N; $k++) {
    Write-Output ("=== route {0} run {1}/{2} ({3}) ===" -f $RouteId, $k, $N, (Get-Date -Format HH:mm:ss))
    Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    schtasks /end /tn CarlaServerNewLog | Out-Null   # clear the 'Running' wedge, else /run is a silent no-op (gotcha)
    Start-Sleep -Seconds 5
    schtasks /run /tn CarlaServerNewLog | Out-Null
    $up = $false
    for ($i = 1; $i -le 120; $i++) {
      if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
      Start-Sleep -Seconds 1
    }
    if (!$up) { Write-Output "  CARLA port failed, skip"; continue }
    $out = Join-Path $Lead ("outputs\local_evaluation_win\{0}_run{1}" -f $RouteId, $k)
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Set-Location $Lead
    $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/tfv6_resnet34",
      "--routes", "data/benchmark_routes/bench2drive/$RouteId.xml", "--bench2drive",
      "--port", "2000", "--timeout", "900", "--output-dir", $out)
    & $Python @rargs *> (Join-Path $Lead ("outputs\multi_{0}_run{1}.log" -f $RouteId, $k))
    $cp = Join-Path $out "checkpoint_endpoint.json"
    if (Test-Path $cp) {
      $j = Get-Content -Raw $cp | ConvertFrom-Json
      $gr = $j._checkpoint.global_record
      $col = [double]$gr.infractions.collisions_vehicle + [double]$gr.infractions.collisions_pedestrian + [double]$gr.infractions.collisions_layout
      $score = [double]$gr.scores_mean.score_composed
      $failed = if (($score -lt 100) -or ($col -gt 0)) { 1 } else { 0 }
      Write-Output ("  RESULT status={0} score={1} collisions={2} failed={3}" -f $gr.status, $score, $col, $failed)
      "$RouteId,$k,$($gr.status),$score,$col,$($gr.infractions.route_dev),$failed" | Out-File -Append -Encoding utf8 $csv
      $results += [pscustomobject]@{ route = $RouteId; failed = $failed }
    } else {
      Write-Output "  RESULT no checkpoint (infra fail)"
      "$RouteId,$k,NO_CHECKPOINT,,,,NA" | Out-File -Append -Encoding utf8 $csv
    }
  }
}

Write-Output "`n=== EMBODIED FAILURE-RATE (statistical verifier output) ==="
foreach ($item in $plan) {
  $RouteId = $item[0]
  $runs = $results | Where-Object { $_.route -eq $RouteId }
  $n = ($runs | Measure-Object).Count
  $f = ($runs | Where-Object { $_.failed -eq 1 } | Measure-Object).Count
  if ($n -gt 0) {
    Write-Output ("route {0}: FAILED {1}/{2} runs = {3:P0} failure rate  ({4})" -f $RouteId, $f, $n, ($f / $n),
      $(if ($f -eq 0 -or $f -eq $n) { "STABLE" } else { "FLAKY - single-run verification would be misleading" }))
  } else {
    Write-Output ("route {0}: no successful runs (infra)" -f $RouteId)
  }
}
Write-Output "ALL DONE"
