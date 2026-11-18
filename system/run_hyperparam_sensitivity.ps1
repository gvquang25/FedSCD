# ==============================================================================
# 1-CLICK WINDOWS GPU RUNNER: HYPERPARAMETER SENSITIVITY (CIFAR-10, alpha=0.5)
# ==============================================================================
# Fixed baseline configuration matching WandB run c7tds897 (76.66% on CIFAR-10):
#   -lam 0.3 --proto_momentum 0.6 --lamda_ins 0.75
#   --lamda_virt_local 0.3 --lamda_virt_global 0.3 --phi_range 0.1 -nct 0.1
#
# Sweeps 6 parameters x 3 non-default values = 18 runs (or +1 with -IncludeDefault).
# Usage:
#   cd system
#   .\run_hyperparam_sensitivity.ps1
#   .\run_hyperparam_sensitivity.ps1 -OnlyParam lam
#   .\run_hyperparam_sensitivity.ps1 -DryRun
# ==============================================================================

param(
    [string]$Device = "cuda",
    [int]$DeviceId = 0,
    [ValidateSet("", "lam", "ins", "virt", "beta", "rho", "tau")]
    [string]$OnlyParam = "",
    [string]$WandbProject = "FedSCD",
    [string]$WandbEntity = "gvquang25-hanoi-university-of-science-and-technology",
    [string]$WandbGroup = "Cifar10_hyperparam_alpha0.5",
    [switch]$IncludeDefault,
    [switch]$ForceAll,
    [switch]$DryRun,
    [bool]$AutoPlot = $true
)

Set-Location $PSScriptRoot

$PythonExe = "..\.venv\Scripts\python.exe"
if (-Not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$pyArgs = @(
    "-u", "run_hyperparam_sensitivity.py",
    "--device", $Device,
    "--device_id", "$DeviceId",
    "--wandb_project", $WandbProject,
    "--wandb_entity", $WandbEntity,
    "--wandb_group", $WandbGroup
)

if ($OnlyParam -ne "") {
    $pyArgs += @("--only_param", $OnlyParam)
}
if ($IncludeDefault) {
    $pyArgs += "--include_default"
}
if ($ForceAll) {
    $pyArgs += "--force_all"
}
if ($DryRun) {
    $pyArgs += "--dry_run"
}

Write-Host "Using Python: $PythonExe" -ForegroundColor Cyan
& $PythonExe @pyArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    Write-Host "`n[ERROR] Hyperparameter sweep exited with code $ExitCode." -ForegroundColor Red
    exit $ExitCode
}

if ($AutoPlot -and (-Not $DryRun)) {
    Write-Host "`n>>> Fetching WandB results and regenerating Figure 4 (2x3 IEEE grid)..." -ForegroundColor Green
    Push-Location ".."
    & $PythonExe "Figure_Plot\plot_hyperparam_sensitivity.py" `
        --fetch `
        --entity $WandbEntity `
        --project $WandbProject `
        --group $WandbGroup
    $PlotCode = $LASTEXITCODE
    Pop-Location
    if ($PlotCode -ne 0) {
        Write-Host "[WARN] Plotting failed with exit code $PlotCode. Re-run manually: python Figure_Plot\plot_hyperparam_sensitivity.py --fetch" -ForegroundColor Yellow
        exit $PlotCode
    }
}
