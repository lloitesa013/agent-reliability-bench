# E1: dump every valid run of a level to CSV (raw evidence -> repo).  -Tag l0 | l1 | l1seedvary | l2 | l3
param([string]$Tag = "l0")
$root = "C:\lead\outputs\e1_$Tag"
$outCsv = "C:\lead\outputs\e1_${Tag}_runs.csv"
$rows = @()
Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
  $dir = $_.Name
  $cp = Join-Path $_.FullName "checkpoint_endpoint.json"
  if (-not (Test-Path $cp)) { return }
  try { $j = Get-Content -Raw $cp | ConvertFrom-Json } catch { return }
  $gr = $j._checkpoint.global_record
  if (-not $gr -or -not $gr.status) { return }
  $rec = $j._checkpoint.records | Select-Object -First 1
  $inf = $gr.infractions
  $parts = $dir -split "_run"
  $rows += [pscustomobject]@{
    level          = $Tag
    route          = $parts[0]
    run            = [int]$parts[1]
    status         = $gr.status
    score_composed = [double]$gr.scores_mean.score_composed
    score_route    = [double]$gr.scores_mean.score_route
    score_penalty  = [double]$gr.scores_mean.score_penalty
    col_vehicle    = [double]$inf.collisions_vehicle
    col_pedestrian = [double]$inf.collisions_pedestrian
    col_layout     = [double]$inf.collisions_layout
    red_light      = [double]$inf.red_light
    stop_infr      = [double]$inf.stop_infraction
    outside_lane   = [double]$inf.outside_route_lanes
    route_dev      = [double]$inf.route_dev
    min_speed      = [double]$inf.min_speed_infractions
    agent_blocked  = [double]$inf.vehicle_blocked
    duration_game  = $(if ($rec) { [double]$rec.meta.duration_game } else { "" })
    duration_sys   = $(if ($rec) { [double]$rec.meta.duration_system } else { "" })
    route_len      = $(if ($rec) { [double]$rec.meta.route_length } else { "" })
  }
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 $outCsv
Write-Output ("wrote {0} rows -> {1}" -f $rows.Count, $outCsv)
