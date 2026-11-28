param(
    [Parameter(Mandatory=$true)][string]$Campaign,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw "Choose a new VM telemetry path" }
$parentDir = Split-Path -Parent $Output
if ($parentDir) { [IO.Directory]::CreateDirectory($parentDir) | Out-Null }
$statePath = Join-Path $Campaign 'state.json'
$timer = [Diagnostics.Stopwatch]::StartNew()
$writer = [IO.StreamWriter]::new($Output, $false)
try {
    while ($timer.Elapsed.TotalSeconds -lt 28200) {
        $phase = 'pilot'
        $status = 'running'
        if (Test-Path -LiteralPath $statePath) {
            $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
            $phase = $state.condition
            $status = $state.status
        }
        $vm = Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue
        $workingSet = if ($null -ne $vm) { $vm.WorkingSet64 } else { $null }
        $writer.WriteLine((@{
            utc = [DateTime]::UtcNow.ToString('o')
            elapsed_s = $timer.Elapsed.TotalSeconds
            phase = $phase
            status = $status
            all_wsl_vm_working_set_bytes = $workingSet
        } | ConvertTo-Json -Compress))
        $writer.Flush()
        if ($status -in @('complete','failed','budget_stopped')) { break }
        Start-Sleep -Seconds 2
    }
} finally {
    $writer.Dispose()
}
