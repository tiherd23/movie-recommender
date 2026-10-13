# Đo xung nhịp, công suất và lý do bị giới hạn của GPU TRONG LÚC chạy tải nặng (scripts/gpu_check.py).
# Chạy từ thư mục đồ án, đã bật .venv:   .\scripts\gpu_monitor.ps1
# Kết quả chi tiết: logs\gpu_monitor.csv (mỗi giây một dòng)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root "logs\gpu_monitor.csv"
New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null

$fields = "timestamp,pstate,clocks.gr,clocks.max.gr,clocks.mem,clocks.max.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active"
Write-Host "Thông tin GPU lúc nghỉ:"
nvidia-smi --query-gpu=name,driver_version,$fields --format=csv

# nvidia-smi ghi log mỗi giây ở tiến trình nền
$mon = Start-Process nvidia-smi -ArgumentList "--query-gpu=$fields", "--format=csv,nounits", "-l", "1", "-f", $log -PassThru -WindowStyle Hidden
try {
    Write-Host "`nĐang chạy gpu_check.py (khoảng 1-2 phút)..."
    python (Join-Path $root "scripts\gpu_check.py")
}
finally {
    Stop-Process -Id $mon.Id -ErrorAction SilentlyContinue
}

# Tóm tắt các giây GPU thật sự bận (utilization > 50%)
$rows = Import-Csv $log | ForEach-Object {
    $o = @{}
    foreach ($p in $_.PSObject.Properties) {
        # tên cột khác nhau giữa các bản driver (clocks.gr / clocks.current.graphics, throttle / event reasons)
        $name = ($p.Name -replace '\s*\[.*\]', '').Trim()
        $name = $name -replace '^clocks\.current\.graphics$', 'clocks.gr' -replace '^clocks\.max\.graphics$', 'clocks.max.gr' `
                      -replace '^clocks\.current\.memory$', 'clocks.mem' -replace '^clocks\.max\.memory$', 'clocks.max.mem' `
                      -replace '^clocks_event_reasons\.active$', 'clocks_throttle_reasons.active'
        $o[$name] = "$($p.Value)".Trim()
    }
    [pscustomobject]$o
}
$busy = $rows | Where-Object { [double]$_.'utilization.gpu' -gt 50 }
if (-not $busy) { Write-Host "Không có giây nào GPU bận > 50%."; return }
$stat = { param($col) ($busy | ForEach-Object { [double]$_.$col } | Measure-Object -Average -Maximum) }
$g = & $stat 'clocks.gr'; $m = & $stat 'clocks.mem'; $p = & $stat 'power.draw'; $t = & $stat 'temperature.gpu'
Write-Host "`n===== TÓM TẮT KHI GPU BẬN ($($busy.Count) giây) ====="
"Xung nhân   : TB {0:N0} MHz, cao nhất {1:N0} MHz (tối đa cho phép {2} MHz)" -f $g.Average, $g.Maximum, $busy[0].'clocks.max.gr'
"Xung bộ nhớ : TB {0:N0} MHz, cao nhất {1:N0} MHz (tối đa cho phép {2} MHz)" -f $m.Average, $m.Maximum, $busy[0].'clocks.max.mem'
"Công suất   : TB {0:N1} W, cao nhất {1:N1} W (giới hạn {2} W)" -f $p.Average, $p.Maximum, $busy[0].'power.limit'
"Nhiệt độ    : TB {0:N0} °C, cao nhất {1:N0} °C" -f $t.Average, $t.Maximum
"P-state     : " + (($busy | Group-Object pstate | ForEach-Object { "$($_.Name) x$($_.Count)" }) -join ", ")
"Lý do giới hạn xung (mã thường gặp nhất): " + (($busy | Group-Object 'clocks_throttle_reasons.active' | Sort-Object Count -Descending | Select-Object -First 3 | ForEach-Object { "$($_.Name) x$($_.Count)" }) -join ", ")
Write-Host "Chi tiết: $log"
