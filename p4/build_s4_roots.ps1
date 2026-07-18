# S4-(3): build junction-based dataset roots (no data copied; route-dir-level junctions).
#   vsi_broad_t  = A2 root + quarantine targeted (10 scenario dirs, EnterActorFlow excluded)
#   vsi_narrow_t = K1-narrow routes {11755,18252,3436,2509,2513} from A2 root + targeted dirs
$Main = "C:\lead\data\carla_leaderboard2_vsi\data"
$Quar = "C:\lead\data\vsi_quarantine"
$NarrowRoutes = @("_11755_", "_18252_", "_3436_", "_2509_", "_2513_")

function Add-Junctions([string]$SrcScenDir, [string]$DstRoot, [string[]]$RouteFilter) {
  $scen = Split-Path $SrcScenDir -Leaf
  $dst = Join-Path $DstRoot $scen
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Get-ChildItem $SrcScenDir -Directory | ForEach-Object {
    $name = "_" + $_.Name + "_"
    if ($RouteFilter) {
      $hit = $false
      foreach ($rf in $RouteFilter) { if ($name -like ("*" + $rf + "*")) { $hit = $true; break } }
      if (-not $hit) { return }
    }
    $link = Join-Path $dst $_.Name
    if (-not (Test-Path $link)) { cmd /c mklink /J "$link" "$($_.FullName)" | Out-Null; $script:n++ }
  }
}

foreach ($root in "C:\lead\data\vsi_broad_t\data", "C:\lead\data\vsi_narrow_t\data") {
  if (Test-Path $root) { Remove-Item -Recurse -Force (Split-Path $root -Parent) }
  New-Item -ItemType Directory -Force -Path $root | Out-Null
}
$n = 0
# broad_t: all main dirs + targeted quarantine dirs
Get-ChildItem $Main -Directory | ForEach-Object { Add-Junctions $_.FullName "C:\lead\data\vsi_broad_t\data" $null }
Get-ChildItem $Quar -Directory | Where-Object { $_.Name -ne "EnterActorFlow" } | ForEach-Object { Add-Junctions $_.FullName "C:\lead\data\vsi_broad_t\data" $null }
Write-Output ("broad_t junctions: " + $n)
$n = 0
# narrow_t: K1 routes from main + all targeted quarantine dirs
Get-ChildItem $Main -Directory | ForEach-Object { Add-Junctions $_.FullName "C:\lead\data\vsi_narrow_t\data" $NarrowRoutes }
Get-ChildItem $Quar -Directory | Where-Object { $_.Name -ne "EnterActorFlow" } | ForEach-Object { Add-Junctions $_.FullName "C:\lead\data\vsi_narrow_t\data" $null }
Write-Output ("narrow_t junctions: " + $n)
# drop empty scenario dirs in narrow_t (scenarios with no matching routes)
Get-ChildItem "C:\lead\data\vsi_narrow_t\data" -Directory | Where-Object { (Get-ChildItem $_.FullName -Directory).Count -eq 0 } | Remove-Item -Force
Write-Output ("broad_t scen dirs: " + (Get-ChildItem C:\lead\data\vsi_broad_t\data -Directory).Count)
Write-Output ("narrow_t scen dirs: " + (Get-ChildItem C:\lead\data\vsi_narrow_t\data -Directory).Count)
