# Chạy một lệnh Python trong môi trường ảo của project và lưu toàn bộ phần in ra vào logs\last.log.
# Cách dùng:  .\run.ps1 scripts/eval_semantic.py
#             .\run.ps1 scripts/eval_semantic.py --device cpu
#             .\run.ps1 -m pytest -q
Set-Location $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs" | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
"=== python $args ===  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath "$PSScriptRoot\logs\last.log"
python @args 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$PSScriptRoot\logs\last.log" -Append
"=== KET THUC, ma thoat $LASTEXITCODE ===  $(Get-Date -Format 'HH:mm:ss')" | Tee-Object -FilePath "$PSScriptRoot\logs\last.log" -Append
