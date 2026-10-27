# Reproduce every result in results/ from scratch (about 5 hours on an RTX 3050 4 GB).
# Usage:  .\run_all.ps1      (from the project folder, with the Ollama app running)
# Failures are detected via exit codes: in Windows PowerShell 5.1, ErrorActionPreference=Stop
# would treat harmless stderr output (Python warnings, Ollama spinners) as fatal.
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force outputs\logs | Out-Null

function Step($name, [string[]]$cmd) {
    Write-Host "==> $name"
    & $py @cmd *> "outputs\logs\$name.log"
    if ($LASTEXITCODE -ne 0) { throw "$name failed, see outputs\logs\$name.log" }
}

Step "test" @("-m", "pytest", "-q", "tests")
Step "prepare_data" @("scripts\01_prepare_data.py")

ollama stop gemma3:4b *> $null   # free the GPU for training
Step "eval_base" @("scripts\03_evaluate.py", "--model", "base")

# Each adapter is evaluated right after it is trained, so early results show up while later runs train.
foreach ($n in 1000, 2000, 4000, 8000) {
    Step "train_n$n" @("scripts\02_train.py", "--train-size", "$n")
    Step "eval_adapter_n$n" @("scripts\03_evaluate.py", "--model", "adapter:n$n")
}

Step "eval_ollama_gemma3_4b" @("scripts\03_evaluate.py", "--model", "ollama:gemma3:4b")
ollama stop gemma3:4b *> $null

Step "export" @("scripts\05_export_to_ollama.py")
Step "eval_ollama_sql-qwen" @("scripts\03_evaluate.py", "--model", "ollama:sql-qwen")
Step "report" @("scripts\04_report.py")
Write-Host "Done. See results\RESULTS.md"
