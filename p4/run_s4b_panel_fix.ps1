# S4-(3) blind-arm panel stage 1 (11755 fix arm) with live curtailment. Usage:
#   run_s4b_panel_fix.ps1 -Name s4b1
# Stops at fails>=3 (REJECT) or when even all-remaining-fail stays <=2 (PASS).
param([string]$Name)
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
$Route = "11755"
$Ck = "outputs/checkpoints/vsi_" + $Name
$csv = Join-Path $Lead ("outputs\{0}_panel_fix.csv" -f $Name)
if (-not (Test-Path $csv)) { "route,run,status,score,collisions,failed,watchdog" | Out-File -Encoding utf8 $csv }
$fails = 0; $passes = 0; $used = 0

for ($k = 1; $k -le 8; $k++) {
  Write-Output ("=== {0} fix run {1}/8 (F{2}/P{3}) {4} ===" -f $Name, $k, $fails, $passes, (Get-Date -Format HH:mm:ss))
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
  $out = Join-Path $Lead ("outputs\local_evaluation_win\{0}_fix_r{1}" -f $Name, $k)
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $log = Join-Path $Lead ("outputs\{0}_fix_r{1}.log" -f $Name, $k)
  $rargs = @("-m", "lead", "--checkpoint", $Ck,
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
    if ((-not $gr.status) -and ($score -eq 0) -and ($col -eq 0)) {
      "$Route,$k,SETUP_CRASH,,,NA,0" | Out-File -Append -Encoding utf8 $csv
      continue
    }
    $failed = if (($score -lt 100) -or ($col -gt 0)) { 1 } else { 0 }
    "$Route,$k,$($gr.status),$score,$col,$failed,0" | Out-File -Append -Encoding utf8 $csv
    $used = $k
    if ($failed -eq 1) { $fails++ } else { $passes++ }
    if ($fails -ge 3) { Write-Output ("CURTAILED: fix REJECT after {0} runs" -f $used); break }
    if (($fails + (8 - $used)) -le 2) { Write-Output ("CURTAILED: fix PASS after {0} runs" -f $used); break }
  } else {
    "$Route,$k,INVALID,,,NA,$wd" | Out-File -Append -Encoding utf8 $csv
  }
}
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output ("{0} PANEL FIX DONE: fails={1} passes={2} used={3}" -f $Name, $fails, $passes, $used)
