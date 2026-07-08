param([string]$RouteId = "11755")
$ErrorActionPreference = "Stop"
$Python = "$env:USERPROFILE\miniconda3\envs\lead-win\python.exe"
$EnvDir = "$env:USERPROFILE\miniconda3\envs\lead-win"
$Lead = "C:\lead"
$CarlaRoot = "C:\CARLA_0.9.15\WindowsNoEditor"
$Log = "$env:USERPROFILE\lead_win_$RouteId.log"
$OutputDir = Join-Path $Lead "outputs\local_evaluation_win\${RouteId}_verify"

$env:CARLA_ROOT = $CarlaRoot
$env:LEAD_PROJECT_ROOT = $Lead
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PATH = "$EnvDir;$EnvDir\Scripts;$EnvDir\Library\bin;$EnvDir\Library\usr\bin;$env:PATH"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

Write-Output "== clean old CARLA =="
Get-Process CarlaUE4 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

Write-Output "== start CARLA =="
schtasks /run /tn CarlaServerNewLog | Out-String | Write-Output

Write-Output "== wait for port 2000 =="
$up = $false
for ($i = 1; $i -le 120; $i++) {
  $conn = Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue
  if ($conn.TcpTestSucceeded) { Write-Output "port up after ${i}s"; $up = $true; break }
  Start-Sleep -Seconds 1
}
if (!$up) { throw "CARLA port 2000 did not open" }

Set-Location -LiteralPath $Lead
if (Test-Path -LiteralPath $OutputDir) { Remove-Item -Recurse -Force -LiteralPath $OutputDir }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Output "== run LEAD route $RouteId =="
$rargs = @("-m","lead","--checkpoint","outputs/checkpoints/tfv6_resnet34",
  "--routes","data/benchmark_routes/bench2drive/$RouteId.xml","--bench2drive",
  "--port","2000","--timeout","900","--output-dir",$OutputDir)
$ErrorActionPreference = "Continue"
& $Python @rargs *> $Log
Write-Output "EXIT $LASTEXITCODE ; log=$Log ; out=$OutputDir"
