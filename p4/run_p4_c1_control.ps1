# P4 recipe 1: control-probe on 28330 (prescriber's move), with LIVE CURTAILMENT.
# LEAD_THROTTLE_SCALE=0.7 on the baseline student; fix line <=2/8 fails; the arm STOPS the moment
# the verdict is mathematically decided (fails>=3 -> REJECT). Fixed-N counterfactual = 8.
$ErrorActionPreference = "Continue"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$env:CARLA_ROOT = "C:\CARLA_0.9.15\WindowsNoEditor"
$env:LEAD_PROJECT_ROOT = $Lead
$env:LEAD_THROTTLE_SCALE = "0.7"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$WALL_LIMIT_MS = 1800000
$Route = "28330"
$csv = Join-Path $Lead "outputs\p4_c1.csv"
if (-not (Test-Path $csv)) { "route,run,status,score,collisions,failed,watchdog" | Out-File -Encoding utf8 $csv }
$fails = 0
$used = 0

for ($k = 1; $k -le 8; $k++) {
  Write-Output ("=== c1 control run {0}/8 (fails so far {1}) {2} ===" -f $k, $fails, (Get-Date -Format HH:mm:ss))
  Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  schtasks /end /tn CarlaServerNewLog | Out-Null
  Start-Sleep -Seconds 5
  schtasks /run /tn CarlaServerNewLog | Out-Null
  $up = $false
  for ($i = 1; $i -le 120; $i++) {
    if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
    Start-Sleep -Seconds 1
  }
  if (!$up) { "$Route,$k,PORT_FAIL,,,NA,0" | Out-File -Append -Encoding utf8 $csv; continue }
  $out = Join-Path $Lead ("outputs\local_evaluation_win\p4c1_{0}_r{1}" -f $Route, $k)
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $log = Join-Path $Lead ("outputs\p4c1_{0}_r{1}.log" -f $Route, $k)
  $rargs = @("-m", "lead", "--checkpoint", "outputs/checkpoints/tfv6_resnet34",
    "--routes", "data/benchmark_routes/bench2drive/$Route.xml", "--bench2drive",
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
    $failed = if (($score -lt 100) -or ($col -gt 0)) { 1 } else { 0 }
    "$Route,$k,$($gr.status),$score,$col,$failed,0" | Out-File -Append -Encoding utf8 $csv
    $used = $k
    if ($failed -eq 1) { $fails++ }
    if ($fails -ge 3) {
      Write-Output ("CURTAILED: verdict decided (REJECT) after {0} runs; fixed-N would use 8" -f $used)
      break
    }
  } else {
    "$Route,$k,INVALID,,,NA,$wd" | Out-File -Append -Encoding utf8 $csv
  }
}
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output ("P4 C1 DONE: fails={0} used={1}" -f $fails, $used)
