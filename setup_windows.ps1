param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-Python311 {
    $candidates = @(
        @{ Command = "C:\Program Files\PyManager\py.exe"; Args = @("-3.11") },
        @{ Command = "$env:LOCALAPPDATA\Python\pythoncore-3.11-64\python.exe"; Args = @() },
        @{ Command = "C:\Program Files\PyManager\python.exe"; Args = @() },
        @{ Command = "py"; Args = @("-3.11") },
        @{ Command = "python"; Args = @() },
        @{ Command = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $command = $candidate.Command
        $args = $candidate.Args

        try {
            if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
                continue
            }

            $versionOutput = @(& $command @args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); print(sys.executable)" 2>$null)
            if ($LASTEXITCODE -ne 0 -or $null -eq $versionOutput -or $versionOutput.Length -lt 2) {
                continue
            }

            $version = [string]$versionOutput[0]
            $executable = [string]$versionOutput[1]
            if ($version -like "3.11.*") {
                return @{
                    Command = $command
                    Args = $args
                    Version = $version
                    Executable = $executable
                }
            }
        }
        catch {
            continue
        }
    }

    throw "Python 3.11 nao encontrado. Instale pelo site https://www.python.org/downloads/release/python-3119/ ou via winget: winget install Python.Python.3.11"
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Step "Localizando Python 3.11"
$python = Resolve-Python311
Write-Host "Python encontrado: $($python.Version)"
Write-Host "Executavel: $($python.Executable)"

Write-Step "Removendo venv antigo, se existir"
if (Test-Path ".\venv") {
    Remove-Item -Recurse -Force ".\venv"
    Write-Host "venv antigo removido."
}
else {
    Write-Host "Nenhum venv existente encontrado."
}

Write-Step "Criando novo venv"
$venvArgs = @($python.Args) + @("-m", "venv", "venv")
& $python.Command @venvArgs
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao criar o venv."
}

$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Python do venv nao foi encontrado em $venvPython"
}

Write-Step "Atualizando pip"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao atualizar pip."
}

if (-not $SkipInstall) {
    Write-Step "Instalando requirements.txt"
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar dependencias."
    }
}
else {
    Write-Step "Instalacao de dependencias ignorada por -SkipInstall"
}

Write-Step "Validando imports principais"
& $venvPython -c "import numpy; import cv2; import mediapipe; import torch; print('numpy', numpy.__version__); print('cv2', cv2.__version__); print('mediapipe', mediapipe.__version__); print('torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao validar imports principais."
}

Write-Step "Ambiente pronto"
Write-Host "Comandos do pipeline:"
Write-Host ".\venv\Scripts\python.exe -m scripts.capture_data --label oi --samples 20 --duration 3"
Write-Host ".\venv\Scripts\python.exe -m scripts.extract_landmarks"
Write-Host ".\venv\Scripts\python.exe -m scripts.train_lstm"
Write-Host ".\venv\Scripts\python.exe -m scripts.evaluate_model"
Write-Host ".\venv\Scripts\python.exe -m scripts.run_capture_inference"
