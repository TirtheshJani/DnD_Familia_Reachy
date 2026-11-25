<#
.SYNOPSIS
    Installs and starts the 0-VRAM Windows Wake Daemon for Reachy Mini's reachy_familiar app.
.DESCRIPTION
    Registers a current-user Windows Scheduled Task (AtLogOn) running familiar_brain.wake_daemon
    so clicking Play on reachy_familiar inside the Reachy Mini Desktop App/Dashboard wakes your
    RTX 4080 WSL2 stack automatically and clicking Stop puts it back to sleep (0 MiB VRAM).
    Also checks Windows and Hyper-V firewall rules for ports 8090, 8091, and UDP 8092.
#>
[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$NoStart,
    [switch]$ConfigureFirewall,
    [string]$Distro = "Ubuntu-24.04",
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$TaskName = "RockyFamiliarWakeDaemon"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "python*" -and $_.CommandLine -like "*familiar_brain.wake_daemon*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "Uninstalled $TaskName and stopped running Wake Daemon." -ForegroundColor Yellow
    exit 0
}

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command pythonw.exe, python.exe -ErrorAction Stop | Select-Object -First 1).Source
}

Write-Host "=== Installing Rocky Wake Daemon ===" -ForegroundColor Cyan
Write-Host "Repo:   $RepoRoot"
Write-Host "Python: $PythonExe"

if ($ConfigureFirewall) {
    Write-Host "=== Configuring Windows & Hyper-V Firewall Rules ===" -ForegroundColor Cyan
    if (-not (Get-NetFirewallRule -DisplayName "Rocky Wake Daemon TCP 8091" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName "Rocky Wake Daemon TCP 8091" -Direction Inbound -Protocol TCP -LocalPort 8091 -Action Allow | Out-Null
        Write-Host "[fw] Added Rocky Wake Daemon TCP 8091" -ForegroundColor Green
    } else {
        Write-Host "[fw] Rocky Wake Daemon TCP 8091 already exists" -ForegroundColor Green
    }
    if (-not (Get-NetFirewallRule -DisplayName "Rocky Wake Discovery UDP 8092" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName "Rocky Wake Discovery UDP 8092" -Direction Inbound -Protocol UDP -LocalPort 8092 -Action Allow | Out-Null
        Write-Host "[fw] Added Rocky Wake Discovery UDP 8092" -ForegroundColor Green
    } else {
        Write-Host "[fw] Rocky Wake Discovery UDP 8092 already exists" -ForegroundColor Green
    }
    if (-not (Get-NetFirewallHyperVRule -Name "WSL-brain-8090" -ErrorAction SilentlyContinue)) {
        New-NetFirewallHyperVRule -Name "WSL-brain-8090" -DisplayName "familiar-brain 8090 (WSL)" -Direction Inbound -VMCreatorId "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}" -Protocol TCP -LocalPorts 8090 -Action Allow | Out-Null
        Write-Host "[fw] Added Hyper-V rule WSL-brain-8090" -ForegroundColor Green
    } else {
        Write-Host "[fw] Hyper-V rule WSL-brain-8090 already exists" -ForegroundColor Green
    }
}

$action = New-ScheduledTaskAction -Execute $PythonExe -Argument "-m familiar_brain.wake_daemon --repo `"$RepoRoot`" --distro `"$Distro`"" -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 0)
$principalArgs = @{}
if ($RunAsUser) {
    $principalArgs["Principal"] = New-ScheduledTaskPrincipal -UserId $RunAsUser -LogonType Interactive -RunLevel Limited
}

$taskRegistered = $false
try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings @principalArgs -Description "0-VRAM Wake Daemon for Reachy Mini D&D Familiar (Rocky)" -Force | Out-Null
    Write-Host "[task] Registered logon Scheduled Task '$TaskName'" -ForegroundColor Green
    $taskRegistered = $true
} catch {
    Write-Host "[task] Could not register Scheduled Task (using startup folder fallback): $($_.Exception.Message)" -ForegroundColor Yellow
    $startupDir = [Environment]::GetFolderPath("Startup")
    $vbs = Join-Path $startupDir "RockyWakeDaemon.vbs"
    $q = '""'
    $vbsCmd = "${q}${PythonExe}${q} -m familiar_brain.wake_daemon --repo ${q}${RepoRoot}${q} --distro ${q}${Distro}${q}"
    Set-Content -Path $vbs -Value "CreateObject(""WScript.Shell"").Run ""$vbsCmd"", 0, False" -Encoding ASCII
    Write-Host "[startup] Wrote hidden logon launcher at $vbs" -ForegroundColor Green
}

if (-not $NoStart) {
    $existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*familiar_brain.wake_daemon*" }
    if (-not $existing) {
        if ($taskRegistered) {
            Start-ScheduledTask -TaskName $TaskName
        } else {
            Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
                CommandLine = "`"$PythonExe`" -m familiar_brain.wake_daemon --repo `"$RepoRoot`" --distro `"$Distro`""
                CurrentDirectory = $RepoRoot
            } | Out-Null
        }
        Start-Sleep -Seconds 2
    }
    try {
        $st = Invoke-RestMethod -Uri "http://127.0.0.1:8091/status" -TimeoutSec 3
        Write-Host "[wake] Daemon online at http://$($st.host):8091 (state=$($st.state))" -ForegroundColor Green
    } catch {
        Write-Host "[wake] Started daemon; check http://127.0.0.1:8091/status" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Firewall check (run in Administrator PowerShell if Reachy or your phone cannot connect):" -ForegroundColor Yellow
Write-Host "  New-NetFirewallRule -DisplayName 'Rocky Wake Daemon TCP 8091' -Direction Inbound -Protocol TCP -LocalPort 8091 -Action Allow"
Write-Host "  New-NetFirewallRule -DisplayName 'Rocky Wake Discovery UDP 8092' -Direction Inbound -Protocol UDP -LocalPort 8092 -Action Allow"
Write-Host "  New-NetFirewallHyperVRule -Name 'WSL-brain-8090' -DisplayName 'familiar-brain 8090 (WSL)' -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8090 -Action Allow"
