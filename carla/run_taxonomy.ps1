$files = Get-ChildItem -Recurse -Filter checkpoint_endpoint.json 'C:\lead\outputs\local_evaluation_win' |
    Where-Object { $_.FullName -notmatch 'native_test|_old_' } |
    ForEach-Object { $_.FullName }
Write-Host ("input route results: " + $files.Count)
& 'C:\Users\tulpa\miniconda3\envs\lead-win\python.exe' 'C:\lead\ops\build_pdmlite_failure_taxonomy.py' `
    --input @files --output 'C:\lead\outputs\taxonomy.md' --json-output 'C:\lead\outputs\taxonomy.json' --pretty
Write-Host "DONE"
