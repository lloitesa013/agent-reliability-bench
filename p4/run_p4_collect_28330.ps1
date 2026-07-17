# P4 expert probe (sealed in P4_PREREG.md): does a TEACHER exist for each candidate target?
# Privileged expert x2 on each of {17655, 26401, 27018, 28330}; clean = score 100, zero collisions.
$ErrorActionPreference = "Continue"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$WALL_LIMIT_MS = 1500000
$Routes = @("28330")
$N = 10
$csv = Join-Path $Lead "outputs\p4_expert_28330.csv"
if (-not (Test-Path $csv)) { "route,run,status,score,collisions,files,watchdog" | Out-File -Encoding utf8 $csv }

foreach ($Route in $Routes) {
  for ($k = 1; $k -le $N; $k++) {
    $out = Join-Path $Lead ("outputs\p4_expert\{0}_e{1}" -f $Route, $k)
    if (Test-Path (Join-Path $out "checkpoint_endpoint.json")) { continue }
    Write-Output ("=== EXPERT probe {0} run {1}/{2} ({3}) ===" -f $Route, $k, $N, (Get-Date -Format HH:mm:ss))
    Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    schtasks /end /tn CarlaServerNewLog | Out-Null
    Start-Sleep -Seconds 5
    schtasks /run /tn CarlaServerNewLog | Out-Null
    $up = $false
    for ($i = 1; $i -le 120; $i++) {
      if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
      Start-Sleep -Seconds 1
    }
    if (!$up) { "$Route,$k,PORT_FAIL,,,,0" | Out-File -Append -Encoding utf8 $csv; continue }
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $Lead ("outputs\p4_expert_{0}_e{1}.log" -f $Route, $k)
    $rargs = @("-m", "lead", "--expert",
      "--routes", "data/benchmark_routes/bench2drive/$Route.xml",
      "--port", "2000", "--timeout", "900", "--output-dir", $out)
    $wd = 0
    $proc = Start-Process -FilePath $Python -ArgumentList $rargs -WorkingDirectory $Lead -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if (-not $proc.WaitForExit($WALL_LIMIT_MS)) {
      $wd = 1
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
      $files = (Get-ChildItem $out -Recurse -File | Measure-Object).Count
      "$Route,$k,$($gr.status),$score,$col,$files,0" | Out-File -Append -Encoding utf8 $csv
    } else {
      "$Route,$k,INVALID,,,,$wd" | Out-File -Append -Encoding utf8 $csv
    }
  }
}
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output "P4 EXPERT PROBE DONE"
