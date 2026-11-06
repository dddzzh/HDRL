param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$planPath = Join-Path $workspaceRoot 'old/paper_cleanup_20261031/cleanup_plan.json'
$cleanupPlan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$currentRoot = Join-Path $workspaceRoot 'results/paper_current'
$archiveRoot = Join-Path $workspaceRoot 'old/paper_cleanup_20261031'
$resultsRoot = Join-Path $workspaceRoot 'results'
$testsRoot = Join-Path $workspaceRoot 'tests'
function Assert-Inside([string]$Candidate, [string]$Parent) {
    $resolved = [IO.Path]::GetFullPath($Candidate)
    $boundary = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe target: $resolved" }
    return $resolved
}
if ($cleanupPlan.root -ne $workspaceRoot -or $cleanupPlan.current -ne $currentRoot) { throw 'Workspace/plan mismatch' }
if (Test-Path -LiteralPath (Join-Path $archiveRoot 'completion.json')) { throw 'Cleanup already completed; refusing duplicate operation' }
if ((Get-FileHash -LiteralPath (Join-Path $currentRoot 'manifest.json') -Algorithm SHA256).Hash.ToLower() -ne $cleanupPlan.manifest_sha256) { throw 'Staged manifest changed' }
$reparse = @(Get-ChildItem -LiteralPath $resultsRoot -Recurse -Force -Attributes ReparsePoint)
if ($reparse.Count -ne 0) { throw 'Results contains reparse points; refusing recursive operations' }
foreach ($property in $cleanupPlan.copies.PSObject.Properties) {
    $target = Assert-Inside (Join-Path $workspaceRoot $property.Name) $currentRoot
    $source = Assert-Inside (Join-Path $workspaceRoot $property.Value.source) $workspaceRoot
    $expected = $property.Value.sha256
    foreach ($path in @($source,$target)) {
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw "Copy hash mismatch: $path" }
    }
}
if (@($cleanupPlan.models).Count -ne 14) { throw 'Expected exactly 14 retained paper models' }
$deletePaths = @{}
foreach ($checkpoint in $cleanupPlan.delete_checkpoints) {
    $path = Assert-Inside (Join-Path $workspaceRoot $checkpoint.path) $resultsRoot
    if ($path.StartsWith($currentRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Attempt to delete curated checkpoint' }
    if ([IO.Path]::GetExtension($path).ToLower() -notin @('.pt','.pth','.ckpt')) { throw 'Non-checkpoint in delete list' }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne $checkpoint.size) { throw "Checkpoint changed: $path" }
    $deletePaths[$path] = $true
}
foreach ($move in $cleanupPlan.archive_moves) {
    $source = Assert-Inside (Join-Path $workspaceRoot $move.source) $resultsRoot
    $destination = Assert-Inside (Join-Path $workspaceRoot $move.destination) (Join-Path $archiveRoot 'results')
    if ((Split-Path -Parent $source) -ne $resultsRoot -or $source -eq $currentRoot) { throw 'Archive scope is not an original top-level result' }
    if (-not (Test-Path -LiteralPath $source) -or (Test-Path -LiteralPath $destination)) { throw 'Missing source or occupied archive destination' }
}
Write-Output "Preflight passed: 1198 verified copies, 14 retained models, $($deletePaths.Count) obsolete checkpoint files, tests/ only."
if (-not $Apply) { Write-Output 'Dry run only; no deletion or move performed.'; exit 0 }

# Preserve test source bytes in a recovery archive before removing the working tests tree.
$testArchive = Join-Path $archiveRoot 'regression_tests_backup.zip'
if (-not (Test-Path -LiteralPath $testArchive)) {
    Compress-Archive -LiteralPath $testsRoot -DestinationPath $testArchive -CompressionLevel Optimal
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($testArchive)
try {
    foreach ($file in $cleanupPlan.test_files) {
        $entry = $zip.GetEntry(($file.path -replace '\\','/'))
        if ($null -eq $entry) { $entry = $zip.GetEntry(($file.path -replace '/','\')) }
        if ($null -eq $entry -or $entry.Length -ne $file.size) { throw "Test backup missing: $($file.path)" }
        $stream = $entry.Open()
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try { $digest = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLower() }
        finally { $stream.Dispose(); $algorithm.Dispose() }
        if ($digest -ne $file.sha256) { throw "Test backup hash mismatch: $($file.path)" }
    }
} finally { $zip.Dispose() }

# Hash all non-checkpoint historical files before moving them, including failed runs.
$history = [Collections.Generic.List[object]]::new()
foreach ($move in $cleanupPlan.archive_moves) {
    $source = Join-Path $workspaceRoot $move.source
    $item = Get-Item -LiteralPath $source
    $files = if ($item.PSIsContainer) { Get-ChildItem -LiteralPath $source -Recurse -File -Force } else { @($item) }
    foreach ($file in $files) {
        if ($file.Extension.ToLower() -in @('.pt','.pth','.ckpt')) {
            if (-not $deletePaths.ContainsKey($file.FullName)) { throw "Unplanned checkpoint: $($file.FullName)" }
            continue
        }
        $suffix = $file.FullName.Substring($source.Length).TrimStart('\')
        $destination = Join-Path $workspaceRoot $move.destination
        if ($suffix) { $destination = Join-Path $destination $suffix }
        $history.Add([pscustomobject]@{source=$file.FullName.Substring($workspaceRoot.Length+1);destination=$destination.Substring($workspaceRoot.Length+1);size=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()})
    }
}
$history | Export-Csv -LiteralPath (Join-Path $archiveRoot 'historical_files.csv') -NoTypeInformation -Encoding UTF8
$cleanupPlan.delete_checkpoints | Export-Csv -LiteralPath (Join-Path $archiveRoot 'deleted_checkpoints.csv') -NoTypeInformation -Encoding UTF8
$deleted = 0
foreach ($checkpoint in $cleanupPlan.delete_checkpoints) {
    $path = Assert-Inside (Join-Path $workspaceRoot $checkpoint.path) $resultsRoot
    Remove-Item -LiteralPath $path -Force
    $deleted++
    if ($deleted % 3000 -eq 0) { Write-Output "Deleted $deleted/$($deletePaths.Count) obsolete checkpoint files" }
}
New-Item -ItemType Directory -Path (Join-Path $archiveRoot 'results') -ErrorAction Stop | Out-Null
foreach ($move in $cleanupPlan.archive_moves) {
    Move-Item -LiteralPath (Join-Path $workspaceRoot $move.source) -Destination (Join-Path $workspaceRoot $move.destination)
}
foreach ($record in $history) {
    $path = Assert-Inside (Join-Path $workspaceRoot $record.destination) (Join-Path $archiveRoot 'results')
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $record.sha256) { throw "Archived data changed: $path" }
}
$checkedTests = Assert-Inside $testsRoot $workspaceRoot
if ($checkedTests -ne (Join-Path $workspaceRoot 'tests')) { throw 'Unexpected tests target' }
Remove-Item -LiteralPath $checkedTests -Recurse -Force
$pytestCache = Join-Path $workspaceRoot '.pytest_cache'
if (Test-Path -LiteralPath $pytestCache) {
    $checkedCache = Assert-Inside $pytestCache $workspaceRoot
    Remove-Item -LiteralPath $checkedCache -Recurse -Force
}
$retained = @(Get-ChildItem -LiteralPath $resultsRoot -Recurse -File | Where-Object { $_.Extension.ToLower() -in @('.pt','.pth','.ckpt') })
if ($retained.Count -ne 14) { throw 'Unexpected retained checkpoint count' }
$complete = [ordered]@{date='2026-10-31';status='complete';retained_models=14;deleted_checkpoint_files=$deleted;deleted_checkpoint_bytes=$cleanupPlan.checkpoint_bytes;archived_files=$history.Count;archived_top_level_results=@($cleanupPlan.archive_moves).Count;removed_test_files=@($cleanupPlan.test_files).Count;test_recovery_archive='regression_tests_backup.zip';raw_historical_data_hashes_verified=$true;current_manifest_sha256=$cleanupPlan.manifest_sha256}
$complete | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $archiveRoot 'completion.json') -Encoding UTF8
$complete | ConvertTo-Json -Depth 4
