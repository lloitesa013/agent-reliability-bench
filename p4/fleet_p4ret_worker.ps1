# P4-c2 PANEL stage 2 (retention arm) fleet worker with live curtailment.
# cand=vsi_p4c2 on reg12 routes; pooled line <=30% of 24 -> GLOBAL FAIL decided at 8 fails.
# Before each job, counts fails across ALL ports' CSVs and exits early if the verdict is decided.
param([int]$Port, [string]$JobFile)
$ErrorActionPreference = "Continue"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:SCRATCH = "C:\lead\tmp_cache"; $env:USER = "tulpa"; $env:USERNAME = "tulpa"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$WALL_LIMIT_MS = 1800000
$task = "CarlaFleet$Port"
$csv = Join-Path $Lead ("outputs\p4ret_p{0}.csv" -f $Port)
if (-not (Test-Path $csv)) { "route,arm,k,status,score,collisions,failed,watchdog" | Out-File -Encoding utf8 $csv }
$jobs = Get-Content $JobFile | Where-Object { $_ -match "," }

function Global-Fails {
  $n = 0
  foreach ($p in 2000, 2004, 2008) {
    $f = Join-Path $Lead ("outputs\p4ret_p{0}.csv" -f $p)
    if (Test-Path $f) {
      $n += (Get-Content $f | Where-Object { $_ -match ",1,0$" } | Measure-Object).Count
    }
  }
  return $n
}

function Restart-Carla {
  Get-CimInstance Win32_Process -Filter "Name='CarlaUE4-Win64-Shipping.exe'" | Where-Object { $_.CommandLine -like "*rpc-port=$Port*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  schtasks /end /tn $task 2>$null | Out-Null
  Start-Sleep -Seconds 4
  schtasks /run /tn $task | Out-Null
  for ($i = 1; $i -le 150; $i++) {
    if ((Test-NetConnection 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded) { return $true }
    Start-Sleep -Seconds 1
  }
  return $false
}

foreach ($job in $jobs) {
  if ((Global-Fails) -ge 8) { Write-Output ("[p{0}] GLOBAL FAIL DECIDED (>=8) - curtailing" -f $Port); break }
  $p = $job.Trim() -split ","
  $Route = $p[0]; $k = $p[2]
  $out = Join-Path $Lead ("outputs\local_evaluation_win\p4ret_{0}_c{1}" -f $Route, $k)
  if (Test-Path (Join-Path $out "checkpoint_endpoint.json")) { continue }
  Write-Output ("[p{0}] ret {1} k{2} ({3})" -f $Port, $Route, $k, (Get-Date -Format HH:mm:ss))
  if (-not (Restart-Carla)) { "$Route,cand,$k,PORT_FAIL,,,NA,0" | Out-File -Append -Encoding utf8 $csv; continue }
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $log = Join-Path $Lead ("outputs\p4ret_{0}_c{1}.log" -f $Route, $k)
  $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/vsi_p4c2",
    "--routes", "data/benchmark_routes/bench2drive/$Route.xml", "--bench2drive",
    "--port", "$Port", "--timeout", "900", "--output-dir", $out)
  $wd = 0
  $proc = Start-Process -FilePath $Python -ArgumentList $rargs -WorkingDirectory $Lead -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  if (-not $proc.WaitForExit($WALL_LIMIT_MS)) {
    $wd = 1
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
    Get-CimInstance Win32_Process -Filter "Name='CarlaUE4-Win64-Shipping.exe'" | Where-Object { $_.CommandLine -like "*rpc-port=$Port*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
  }
  $cp = Join-Path $out "checkpoint_endpoint.json"
  if ((-not $wd) -and (Test-Path $cp)) {
    $j = Get-Content -Raw $cp | ConvertFrom-Json
    $gr = $j._checkpoint.global_record
    $col = [double]$gr.infractions.collisions_vehicle + [double]$gr.infractions.collisions_pedestrian + [double]$gr.infractions.collisions_layout
    $score = [double]$gr.scores_mean.score_composed
    if ((-not $gr.status) -and ($score -eq 0) -and ($col -eq 0)) {
      "$Route,cand,$k,SETUP_CRASH,,,NA,0" | Out-File -Append -Encoding utf8 $csv
      continue
    }
    $failed = if (($score -lt 100) -or ($col -gt 0)) { 1 } else { 0 }
    "$Route,cand,$k,$($gr.status),$score,$col,$failed,0" | Out-File -Append -Encoding utf8 $csv
  } else {
    "$Route,cand,$k,INVALID,,,NA,$wd" | Out-File -Append -Encoding utf8 $csv
  }
}
Get-CimInstance Win32_Process -Filter "Name='CarlaUE4-Win64-Shipping.exe'" | Where-Object { $_.CommandLine -like "*rpc-port=$Port*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
schtasks /end /tn $task 2>$null | Out-Null
Write-Output ("[p{0}] P4 RET WORKER DONE" -f $Port)
