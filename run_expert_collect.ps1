# A-1 (retention-DAgger brick 1): collect PDM-Lite/LEAD EXPERT demonstrations on the flaky route family.
# Runs `python -m lead --expert` (datagen defaults ON) on route 11755 xN with the hard watchdog, then
# verifies demonstration data actually landed (file count + size per run). This is the DATA side of an
# ACCEPTED embodied fix: expert demos on the failure scenario + (later) replay data for retention.
$ErrorActionPreference = "Continue"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$WALL_LIMIT_MS = 1500000   # 25 min: expert + datagen is slower than eval
$Route = "11755"
$N = 3
$csv = Join-Path $Lead "outputs\expert_collect_$Route.csv"
"run,status,score,files,mb,watchdog" | Out-File -Encoding utf8 $csv

for ($k = 1; $k -le $N; $k++) {
  Write-Output ("=== EXPERT collect route {0} run {1}/{2} ({3}) ===" -f $Route, $k, $N, (Get-Date -Format HH:mm:ss))
  Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  schtasks /end /tn CarlaServerNewLog | Out-Null
  Start-Sleep -Seconds 5
  schtasks /run /tn CarlaServerNewLog | Out-Null
  $up = $false
  for ($i = 1; $i -le 120; $i++) {
    if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
    Start-Sleep -Seconds 1
  }
  if (!$up) { Write-Output "  CARLA port failed, skip"; "$k,PORT_FAIL,,,,0" | Out-File -Append -Encoding utf8 $csv; continue }
  $out = Join-Path $Lead ("outputs\expert_demos\{0}_e{1}" -f $Route, $k)
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $log = Join-Path $Lead ("outputs\expert_{0}_e{1}.log" -f $Route, $k)
  $rargs = @("-m", "lead", "--expert",
    "--routes", "data/benchmark_routes/bench2drive/$Route.xml",
    "--port", "2000", "--timeout", "900", "--output-dir", $out)
  $wd = 0
  $proc = Start-Process -FilePath $Python -ArgumentList $rargs -WorkingDirectory $Lead -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  if (-not $proc.WaitForExit($WALL_LIMIT_MS)) {
    $wd = 1; Write-Output "  WATCHDOG kill (>25min)"
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
    Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
  }
  # result + did demonstration data land?
  $score = ""; $status = "NO_CHECKPOINT"
  $cp = Join-Path $out "checkpoint_endpoint.json"
  if (Test-Path $cp) {
    $j = Get-Content -Raw $cp | ConvertFrom-Json
    $status = $j._checkpoint.global_record.status
    $score = [double]$j._checkpoint.global_record.scores_mean.score_composed
  }
  $files = (Get-ChildItem -Recurse -File $out -ErrorAction SilentlyContinue | Measure-Object).Count
  $mb = [math]::Round(((Get-ChildItem -Recurse -File $out -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum / 1MB), 1)
  Write-Output ("  RESULT status={0} score={1} data: {2} files, {3} MB (watchdog={4})" -f $status, $score, $files, $mb, $wd)
  "$k,$status,$score,$files,$mb,$wd" | Out-File -Append -Encoding utf8 $csv
}
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output "ALL DONE (expert collect)"
