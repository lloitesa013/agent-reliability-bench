$dirs = Get-ChildItem C:\lead\outputs\local_evaluation_win\ -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '_run\d' } | Sort-Object Name
foreach ($d in $dirs) {
  $cp = Join-Path $d.FullName 'checkpoint_endpoint.json'
  if (Test-Path $cp) {
    try {
      $j = Get-Content -Raw $cp | ConvertFrom-Json
      $gr = $j._checkpoint.global_record
      $inf = $gr.infractions
      $col = [double]$inf.collisions_vehicle + [double]$inf.collisions_pedestrian + [double]$inf.collisions_layout
      $rd = [double]$inf.route_dev
      $verdict = if ($col -gt 0 -or $rd -gt 0 -or [double]$gr.scores_mean.score_composed -lt 60) { 'FAIL' } else { 'pass' }
      Write-Output ("{0,-16} | {1,-30} | score={2,6} | collisions={3} route_dev={4} | {5}" -f $d.Name, $gr.status, $gr.scores_mean.score_composed, $col, $rd, $verdict)
    } catch { Write-Output ("{0,-16} | PARSE_ERR" -f $d.Name) }
  } else { Write-Output ("{0,-16} | (empty / stalled run)" -f $d.Name) }
}
