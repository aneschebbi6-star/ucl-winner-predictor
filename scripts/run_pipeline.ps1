# Pipeline complet : dataset -> entrainement calibre -> prediction finale
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "==> Dataset anti-leakage"
python src/dataset.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Entrainement calibre + artifacts"
python src/model.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Prediction finale"
python src/predict.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Tests pytest"
python -m pytest tests/ -q
exit $LASTEXITCODE
