<#
.SYNOPSIS
    Builds and installs the native `reachy_familiar` ReachyMiniApp onto Reachy Mini.
.DESCRIPTION
    Packages reachy_familiar/ into a Python wheel, attempts installation directly via the
    Reachy Mini daemon REST API (POST /api/apps/install, zero SSH password needed), and
    falls back cleanly to SSH install into /venvs/apps_venv if the daemon rejects a local URL.
    Leaves existing robot apps (rocky, rocky_mini, hand_tracker_v2, reachy_mini_conversation_app) untouched.
#>
[CmdletBinding()]
param(
    [string]$RobotHost = "10.0.0.100",
    [string]$SshUser = "pollen",
    [string]$IdentityFile = "",
    [switch]$ForceSsh
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$AppDir = Join-Path $RepoRoot "reachy_familiar"
$DistDir = Join-Path $AppDir "dist"

Write-Host "=== Building reachy_familiar wheel ===" -ForegroundColor Cyan
if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }

$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

# Build wheel using uv or pip wheel (no network needed if setuptools/build present, else uv build)
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv build --wheel --out-dir $DistDir $AppDir
} else {
    & $Py -m pip wheel --no-deps --wheel-dir $DistDir $AppDir
}
$Wheel = Get-ChildItem -Path $DistDir -Filter "reachy_familiar-*.whl" | Select-Object -First 1
if (-not $Wheel) {
    Write-Error "Failed to build reachy_familiar wheel in $DistDir"
    exit 1
}
Write-Host "[build] Built $($Wheel.Name)" -ForegroundColor Green

# Resolve robot IPv4 cleanly
$ResolvedRobot = $RobotHost
try {
    $addrs = [System.Net.Dns]::GetHostAddresses($RobotHost)
    $ipv4 = $addrs | Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } | Select-Object -First 1
    if ($ipv4) { $ResolvedRobot = $ipv4.IPAddressToString }
} catch {}

# Auto-detect existing passwordless SSH key for Reachy Mini if present
if (-not $IdentityFile) {
    $defaultKey = Join-Path $env:USERPROFILE ".ssh\rocky_reachy_ed25519"
    if (Test-Path $defaultKey) {
        $IdentityFile = $defaultKey
        $ForceSsh = $true
    }
}

$SshOpts = @("-o", "StrictHostKeyChecking=no")
if ($IdentityFile -and (Test-Path $IdentityFile)) {
    $SshOpts += @("-i", $IdentityFile, "-o", "BatchMode=yes")
}

if (-not $ForceSsh) {
    Write-Host "[install] Trying Reachy Mini HTTP API at http://${ResolvedRobot}:8000/api/apps/install ..." -ForegroundColor Cyan
    # Serve the wheel briefly over LAN HTTP so the robot daemon can pull it if supported.
    $lanIp = & $Py -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('$ResolvedRobot',8000)); print(s.getsockname()[0])"
    $httpJob = Start-Job -ScriptBlock {
        param($pyExe, $dir)
        & $pyExe -m http.server 8099 --bind 0.0.0.0 --directory $dir
    } -ArgumentList $Py, $DistDir
    Start-Sleep -Seconds 1
    $httpOk = $false
    try {
        $payload = @{
            name = "reachy_familiar"
            source_kind = "local"
            url = "http://${lanIp}:8099/$($Wheel.Name)"
            description = "Rocky D&D Familiar (Shadows over Nephria)"
        } | ConvertTo-Json
        $resp = Invoke-RestMethod -Method Post -Uri "http://${ResolvedRobot}:8000/api/apps/install" -ContentType "application/json" -Body $payload -TimeoutSec 15
        if ($resp.job_id) {
            Write-Host "[install] Daemon job $($resp.job_id) started; polling..." -ForegroundColor Cyan
            for ($i = 0; $i -lt 60; $i++) {
                Start-Sleep -Seconds 1
                $st = Invoke-RestMethod -Uri "http://${ResolvedRobot}:8000/api/apps/job-status/$($resp.job_id)" -TimeoutSec 5
                if ($st.status -eq "completed") {
                    Write-Host "[install] Installed reachy_familiar via HTTP API!" -ForegroundColor Green
                    $httpOk = $true
                    break
                }
                if ($st.status -eq "failed") {
                    Write-Host "[install] HTTP install job failed: $($st.error)" -ForegroundColor Yellow
                    break
                }
            }
        }
    } catch {
        Write-Host "[install] HTTP install not accepted by daemon ($($_.Exception.Message)); falling back to SSH." -ForegroundColor Yellow
    } finally {
        Stop-Job $httpJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $httpJob -ErrorAction SilentlyContinue | Out-Null
    }
    if ($httpOk) {
        Write-Host "Open Reachy Mini Desktop App or http://${ResolvedRobot}:8000 and click Play on reachy_familiar!" -ForegroundColor Green
        exit 0
    }
}

Write-Host "[ssh] Copying $($Wheel.Name) to ${SshUser}@${ResolvedRobot} and installing into /venvs/apps_venv ..." -ForegroundColor Cyan
& scp @SshOpts $Wheel.FullName "${SshUser}@${ResolvedRobot}:/tmp/$($Wheel.Name)"
if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
& ssh @SshOpts "${SshUser}@${ResolvedRobot}" "/venvs/apps_venv/bin/pip install --no-deps --force-reinstall /tmp/$($Wheel.Name) && rm -f /tmp/$($Wheel.Name)"
if ($LASTEXITCODE -ne 0) { throw "ssh pip install failed with exit code $LASTEXITCODE" }
$metaJson = '{"id":"local/reachy_familiar","author":"local","private":true,"isPythonApp":true,"cardData":{"title":"Rocky D&D Familiar","emoji":"\ud83c\udfb2","colorFrom":"yellow","colorTo":"indigo","short_description":"Shadows over Nephria D&D familiar (wakes RTX 4080 brain + Art Deco Companion UI)"}}'
$metaJson | & ssh @SshOpts "${SshUser}@${ResolvedRobot}" "mkdir -p /venvs/.app_metadata && cat > /venvs/.app_metadata/reachy_familiar.json"
Write-Host "[install] Installed reachy_familiar into /venvs/apps_venv via SSH!" -ForegroundColor Green
Write-Host "Done! Open Reachy Mini Desktop App or http://${ResolvedRobot}:8000 and click Play on reachy_familiar." -ForegroundColor Green
