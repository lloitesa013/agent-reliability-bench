# E1 L0 truth audit: files are ground truth, not the master log.
# VALID = checkpoint_endpoint.json exists AND _checkpoint.global_record.status is non-empty.
$root = "C:\lead\outputs\e1_l0"
$rows = @()
Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
  $name = $_.Name
  $cp = Join-Path $_.FullName "checkpoint_endpoint.json"
  if (-not (Test-Path $cp)) {
    $rows += [pscustomobject]@{ dir = $name; state = "NOFILE"; status = ""; score = ""; col = "" }
    return
  }
  try {
    $j = Get-Content -Raw $cp | ConvertFrom-Json
    $gr = $j._checkpoint.global_record
    if ($gr -and $gr.status) {
      $col = [double]$gr.infractions.collisions_vehicle + [double]$gr.infractions.collisions_pedestrian + [double]$gr.infractions.collisions_layout
      $rows += [pscustomobject]@{ dir = $name; state = "VALID"; status = $gr.status; score = [double]$gr.scores_mean.score_composed; col = $col }
    } else {
      $rows += [pscustomobject]@{ dir = $name; state = "EMPTY"; status = ""; score = ""; col = "" }
    }
  } catch {
    $rows += [pscustomobject]@{ dir = $name; state = "PARSE_ERR"; status = ""; score = ""; col = "" }
  }
}
$rows | Format-Table -AutoSize | Out-String -Width 200
Write-Output "--- per-route VALID counts ---"
foreach ($r in @("11755", "18252", "3436")) {
  $v = @($rows | Where-Object { $_.dir -like ($r + "_run*") -and $_.state -eq "VALID" })
  $f = @($v | Where-Object { $_.score -lt 100 -or $_.col -gt 0 })
  Write-Output ("{0}: valid={1} fails={2}" -f $r, $v.Count, $f.Count)
}
