# A-2 (retention-DAgger brick 2): SCALE expert demonstration collection into a training dataset.
#   FIX set      : flaky-failure routes (11755 EnterActorFlow x8, 18252 pedestrian x8)
#   RETENTION set: stable routes the student already passes (3436/2509/2513 x4 each) - the replay data
#                  that protects existing capability during retraining (VSI-0 R12 lesson).
# Fresh CARLA per run + hard watchdog. Output: C:\lead\outputs\expert_dataset\<route>_r<k>\
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
$jobs = @()
foreach ($r in @("11755", "18252")) { for ($k = 1; $k -le 8; $k++) { $jobs += ,@($r, $k, "fix") } }
foreach ($r in @("3436", "2509", "2513")) { for ($k = 1; $k -le 4; $k++) { $jobs += ,@($r, $k, "retention") } }
$csv = Join-Path $Lead "outputs\expert_dataset\collection.csv"
New-Item -ItemType Directory -Force -Path (Join-Path $Lead "outputs\expert_dataset") | Out-Null
"route,run,set,status,score,files,mb,watchdog" | Out-File -Encoding utf8 $csv
$done = 0

foreach ($job in $jobs) {
  $Route = $job[0]; $k = $job[1]; $set = $job[2]; $done++
  Write-Output ("=== [{0}/{1}] {2} route {3} run {4} ({5}) ===" -f $done, $jobs.Count, $set, $Route, $k, (Get-Date -Format HH:mm:ss))
  Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  schtasks /end /tn CarlaServerNewLog | Out-Null
  Start-Sleep -Seconds 5
  schtasks /run /tn CarlaServerNewLog | Out-Null
  $up = $false
  for ($i = 1; $i -le 120; $i++) {
    if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
    Start-Sleep -Seconds 1
  }
  if (!$up) { Write-Output "  CARLA port failed, skip"; "$Route,$k,$set,PORT_FAIL,,,,0" | Out-File -Append -Encoding utf8 $csv; continue }
  $out = Join-Path $Lead ("outputs\expert_dataset\{0}_r{1}" -f $Route, $k)
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $log = Join-Path $Lead ("outputs\expert_dataset\log_{0}_r{1}.log" -f $Route, $k)
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
  $score = ""; $status = "NO_CHECKPOINT"
  $cp = Join-Path $out "checkpoint_endpoint.json"
  if (Test-Path $cp) {
    $j = Get-Content -Raw $cp | ConvertFrom-Json
    $status = $j._checkpoint.global_record.status
    $score = [double]$j._checkpoint.global_record.scores_mean.score_composed
  }
  $files = (Get-ChildItem -Recurse -File $out -ErrorAction SilentlyContinue | Measure-Object).Count
  $mb = [math]::Round(((Get-ChildItem -Recurse -File $out -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum / 1MB), 1)
  Write-Output ("  RESULT status={0} score={1} data={2} files / {3} MB (wd={4})" -f $status, $score, $files, $mb, $wd)
  "$Route,$k,$set,$status,$score,$files,$mb,$wd" | Out-File -Append -Encoding utf8 $csv
}
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output "`n=== DATASET SUMMARY ==="
Import-Csv $csv | Group-Object set | ForEach-Object {
  $ok = ($_.Group | Where-Object { $_.status -eq "Perfect" -or ([double]::TryParse($_.score, [ref]$null) -and [double]$_.score -ge 100) } | Measure-Object).Count
  Write-Output ("{0}: {1}/{2} clean expert runs" -f $_.Name, $ok, $_.Count)
}
Write-Output "ALL DONE (expert dataset)"
