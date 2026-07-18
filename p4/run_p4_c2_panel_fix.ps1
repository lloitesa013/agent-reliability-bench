# P4-c2 PANEL stage 1 (fix arm) with LIVE CURTAILMENT, arm-sequencing form:
# fix arm runs FIRST; if it decides REJECT the retention arm never runs (the largest streaming save).
# Lines (sealed in p4_trials.jsonl): fix <=2/8. Stop when fails>=3 (REJECT) or passes>=6 (PASS).
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
$Route = "28330"
$Ck = "outputs/checkpoints/vsi_p4c2"
$csv = Join-Path $Lead "outputs\p4_panel_fix.csv"
if (-not (Test-Path $csv)) { "route,run,status,score,collisions,failed,watchdog" | Out-File -Encoding utf8 $csv }
$fails = 0; $passes = 0; $used = 0

for ($k = 1; $k -le 8; $k++) {
  Write-Output ("=== c2 panel fix run {0}/8 (F{1}/P{2}) {3} ===" -f $k, $fails, $passes, (Get-Date -Format HH:mm:ss))
  Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  schtasks /end /tn CarlaServerNewLog | Out-Null
  Start-Sleep -Seconds 5
  schtasks /run /tn CarlaServerNewLog | Out-Null
  $up = $false
  for ($i = 1; $i -le 120; $i++) {
    if ((Test-NetConnection 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded) { $up = $true; break }
    Start-Sleep -Seconds 1
  }
  if (!$up) { "$Route,$k,PORT_FAIL,,,NA,0" | Out-File -Append -Encoding utf8 $csv; continue }
  $out = Join-Path $Lead ("outputs\local_evaluation_win\p4pan_{0}_r{1}" -f $Route, $k)
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $log = Join-Path $Lead ("outputs\p4pan_{0}_r{1}.log" -f $Route, $k)
  $rargs = @("-m", "lead", "--checkpoint", $Ck,
    "--routes", "data/benchmark_routes/bench2drive/$Route.xml", "--bench2drive",
    "--port", "2000", "--timeout", "900", "--output-dir", $out)
  $wd = 0
  $proc = Start-Process -FilePath $Python -ArgumentList $rargs -WorkingDirectory $Lead -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  if (-not $proc.WaitForExit($WALL_LIMIT_MS)) {
    $wd = 1
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
    Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
  }
  $cp = Join-Path $out "checkpoint_endpoint.json"
  if ((-not $wd) -and (Test-Path $cp)) {
    $j = Get-Content -Raw $cp | ConvertFrom-Json
    $gr = $j._checkpoint.global_record
    $col = [double]$gr.infractions.collisions_vehicle + [double]$gr.infractions.collisions_pedestrian + [double]$gr.infractions.collisions_layout
    $score = [double]$gr.scores_mean.score_composed
    # validity guard: empty status + score 0 + zero collisions = harness/setup crash, NOT a driving
    # failure -- record INVALID so an infra fault can never manufacture a verdict
    if ((-not $gr.status) -and ($score -eq 0) -and ($col -eq 0)) {
      "$Route,$k,SETUP_CRASH,,,NA,0" | Out-File -Append -Encoding utf8 $csv
      continue
    }
    $failed = if (($score -lt 100) -or ($col -gt 0)) { 1 } else { 0 }
    "$Route,$k,$($gr.status),$score,$col,$failed,0" | Out-File -Append -Encoding utf8 $csv
    $used = $k
    if ($failed -eq 1) { $fails++ } else { $passes++ }
    if ($fails -ge 3) { Write-Output ("CURTAILED: fix REJECT after {0} runs (fixed-N=8)" -f $used); break }
    if ($passes -ge 6) { Write-Output ("CURTAILED: fix PASS after {0} runs (fixed-N=8)" -f $used); break }
  } else {
    "$Route,$k,INVALID,,,NA,$wd" | Out-File -Append -Encoding utf8 $csv
  }
}
Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
schtasks /end /tn CarlaServerNewLog | Out-Null
Write-Output ("P4 PANEL FIX DONE: fails={0} passes={1} used={2}" -f $fails, $passes, $used)
