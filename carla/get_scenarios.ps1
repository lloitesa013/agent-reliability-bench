foreach ($r in '3905','3436','11755','2201','18252','24224','24041','24041') {
  $f = "C:\lead\data\benchmark_routes\bench2drive\$r.xml"
  if (Test-Path $f) {
    $raw = Get-Content -Raw $f
    $m = [regex]::Match($raw, 'type="([^"]*)"')
    $town = [regex]::Match($raw, 'town="([^"]*)"').Groups[1].Value
    Write-Host ("{0} | town={1} | scenario={2}" -f $r, $town, $m.Groups[1].Value)
  } else { Write-Host "$r -> (no xml)" }
}
