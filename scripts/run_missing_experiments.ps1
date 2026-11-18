# ==============================================================================
# PROXY RUNNER: Chay benchmark 84 thi nghiem tu thu muc scripts/
# ==============================================================================
param(
    [string]$Device = "cuda",
    [int]$DeviceId = 0,
    [int]$GlobalRounds = 100,
    [int]$LocalEpochs = 1,
    [int]$BatchSize = 32,
    [double]$LocalLearningRate = 0.01,
    [int]$NumClients = 20,
    [double]$JoinRatio = 1.0,
    [int]$Times = 1,
    [int]$Seed = 2006,
    [int]$TsneGap = 20,
    [switch]$NoTsne,
    [switch]$ForceAll,
    [string]$WandbProject = "FedSCD",
    [string]$WandbEntity = "gvquang25-hanoi-university-of-science-and-technology"
)

$OriginalDir = Get-Location
try {
    $SystemDir = Join-Path $PSScriptRoot "..\system"
    Set-Location $SystemDir
    & ".\run_missing_experiments.ps1" @PSBoundParameters
}
finally {
    Set-Location $OriginalDir
}
