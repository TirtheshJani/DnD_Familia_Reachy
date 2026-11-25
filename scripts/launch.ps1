<#
.SYNOPSIS
    Windows launcher for the Rocky D&D familiar stack running in WSL2.
.DESCRIPTION
    Cleans stale SSL_CERT_FILE, verifies Hyper-V firewall rules for ports 8765 and 8090,
    verifies WSL2 /mnt/c repo access, optionally resolves reachy-mini.local with IP fallback,
    prints the phone Companion UI URL, and starts scripts/start_all.sh in WSL2.
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$SkipRobot,
    [switch]$AutoSession,
    [switch]$NoAudioTap,
    [string]$RobotHost = "",
    [string]$FallbackRobotIp = "10.0.0.100",
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

Write-Host "=== Rocky Familiar Launcher ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"

# 1. Clean stale SSL_CERT_FILE inherited from old Conda installs
if ($env:SSL_CERT_FILE) {
    if (-not (Test-Path $env:SSL_CERT_FILE)) {
        Write-Host "[env] Removing stale SSL_CERT_FILE ($($env:SSL_CERT_FILE))" -ForegroundColor Yellow
        Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
    } else {
        Write-Host "[env] SSL_CERT_FILE ok: $($env:SSL_CERT_FILE)"
    }
}

# 2. Check Hyper-V firewall rules for ports 8765 (S2S / AudioTap) and 8090 (Companion UI)
$vmCreatorId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
$requiredPorts = @(
    @{ Port = 8765; Name = "WSL-s2s-8765";   Display = "speech-to-speech 8765 (WSL)" },
    @{ Port = 8090; Name = "WSL-brain-8090"; Display = "familiar-brain 8090 (WSL)" }
)

try {
    $hvSetting = Get-NetFirewallHyperVVMSetting -PolicyStore ActiveStore -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $vmCreatorId } | Select-Object -First 1
    $defaultInbound = if ($hvSetting) { "$($hvSetting.DefaultInboundAction)" } else { "Unknown" }
    Write-Host "[firewall] Hyper-V VM DefaultInboundAction = $defaultInbound"

    $hvRules = @(Get-NetFirewallHyperVRule -PolicyStore ActiveStore -ErrorAction SilentlyContinue)
    $missingRules = @()
    foreach ($req in $requiredPorts) {
        $portStr = "$($req.Port)"
        $matched = $hvRules | Where-Object {
            "$($_.Enabled)" -eq "True" -and
            "$($_.Direction)" -eq "Inbound" -and
            "$($_.Action)" -eq "Allow" -and
            "$($_.Protocol)" -eq "TCP" -and
            (@($_.LocalPorts) -contains $portStr -or "$($_.LocalPorts)" -match "\b$portStr\b")
        }
        if ($matched) {
            Write-Host "[firewall] OK TCP $portStr allowed ($($matched[0].Name))" -ForegroundColor Green
        } else {
            Write-Host "[firewall] WARNING: No enabled Hyper-V Inbound Allow rule for TCP $portStr" -ForegroundColor Yellow
            $missingRules += $req
        }
    }

    if ($missingRules.Count -gt 0) {
        Write-Host ""
        Write-Host "To allow LAN devices (phone / Reachy Mini) to reach WSL2, run in an Administrator PowerShell:" -ForegroundColor Yellow
        foreach ($m in $missingRules) {
            Write-Host ("  New-NetFirewallHyperVRule -Name `"{0}`" -DisplayName `"{1}`" -Direction Inbound -VMCreatorId '{2}' -Protocol TCP -LocalPorts {3} -Action Allow" -f $m.Name, $m.Display, $vmCreatorId, $m.Port) -ForegroundColor White
        }
        Write-Host ""
    }
} catch {
    Write-Host "[firewall] Could not inspect Hyper-V firewall rules: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 3. Detect Windows LAN IP for Companion UI URL
function Get-LanIPv4 {
    try {
        $ifMetrics = @{}
        Get-NetIPInterface -AddressFamily IPv4 -ErrorAction SilentlyContinue | ForEach-Object {
            $ifMetrics[[int]$_.InterfaceIndex] = [int]$_.InterfaceMetric
        }
        $upAdapters = @{}
        Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { "$($_.Status)" -eq 'Up' } | ForEach-Object {
            $upAdapters[[int]$_.InterfaceIndex] = $true
        }
        $candidates = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object {
                "$($_.AddressState)" -eq 'Preferred' -and
                $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and
                $_.InterfaceAlias -notmatch 'Loopback|vEthernet|WSL|Docker|Tailscale|Bluetooth' -and
                $_.PrefixOrigin -ne 'WellKnown' -and
                ($upAdapters.Count -eq 0 -or $upAdapters.ContainsKey([int]$_.InterfaceIndex))
            } |
            Select-Object *, @{
                Name = 'TrueMetric'
                Expression = {
                    if ($ifMetrics.ContainsKey([int]$_.InterfaceIndex)) { $ifMetrics[[int]$_.InterfaceIndex] } else { 9999 }
                }
            } |
            Sort-Object -Property TrueMetric, InterfaceIndex
        if ($candidates) {
            return ($candidates | Select-Object -First 1).IPAddress
        }
    } catch {}
    return $null
}

$LanIp = Get-LanIPv4
if (-not $LanIp) { $LanIp = "<host-lan-ip>" }

# 4. Resolve robot host (never probe physical robot during -DryRun or -SkipRobot)
$ResolvedRobot = ""
if ($DryRun -or $SkipRobot) {
    $plannedTarget = if ($RobotHost) { $RobotHost } else { "reachy-mini.local (fallback $FallbackRobotIp)" }
    Write-Host "[robot] Skipping robot network probe (-DryRun/-SkipRobot active; planned target: $plannedTarget)"
} else {
    $candidates = if ($RobotHost) { @($RobotHost, $FallbackRobotIp) } else { @("reachy-mini.local", $FallbackRobotIp) }
    foreach ($cand in ($candidates | Select-Object -Unique)) {
        if (-not $cand) { continue }
        try {
            $addrs = [System.Net.Dns]::GetHostAddresses($cand)
            if (-not $addrs) { continue }
            $ipv4 = ($addrs | Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } | Select-Object -First 1)
            $resolvedTarget = if ($ipv4) { $ipv4.IPAddressToString } else { $cand }
            $resp = Invoke-WebRequest -Uri "http://$($resolvedTarget):8000/api/apps/current-app-status" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($resp.StatusCode -lt 500) {
                $ResolvedRobot = $resolvedTarget
                break
            }
        } catch {
            Write-Host "[robot] Probe failed for ${cand}: $($_.Exception.Message)" -ForegroundColor DarkYellow
        }
    }
    if (-not $ResolvedRobot) {
        $ResolvedRobot = if ($RobotHost) { $RobotHost } else { $FallbackRobotIp }
        Write-Host "[robot] Using fallback host $ResolvedRobot (daemon probe did not answer yet)" -ForegroundColor Yellow
    } else {
        Write-Host "[robot] Resolved robot at $ResolvedRobot" -ForegroundColor Green
    }
}

# 5. Convert Windows repo path to WSL /mnt/<drive>/... and verify accessibility
if ($RepoRoot -match '^([A-Za-z]):\\(.*)$') {
    $drive = $Matches[1].ToLowerInvariant()
    $rest = ($Matches[2] -replace '\\', '/')
    $WslRepo = "/mnt/$drive/$rest"
} else {
    Write-Error "RepoRoot '$RepoRoot' is not a local drive path."
    exit 1
}

Write-Host "[wsl] Checking repo accessibility in '$Distro' at $WslRepo ..."
$null = wsl -d $Distro -e bash -lc "test -f '$WslRepo/scripts/start_all.sh'" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[wsl] ERROR: $WslRepo/scripts/start_all.sh is not visible inside WSL2 '$Distro'." -ForegroundColor Red
    Write-Host "      Verify /etc/wsl.conf has:" -ForegroundColor Yellow
    Write-Host "        [automount]" -ForegroundColor White
    Write-Host "        enabled = true" -ForegroundColor White
    Write-Host "      Then run 'wsl --shutdown' and retry." -ForegroundColor Yellow
    exit 1
}
Write-Host "[wsl] OK Repo accessible at $WslRepo" -ForegroundColor Green

# 6. Build start_all.sh command
$startArgs = @()
if ($SkipRobot)   { $startArgs += "--skip-robot" }
if ($AutoSession) { $startArgs += "--auto-session" }
if ($NoAudioTap)  { $startArgs += "--no-audio-tap" }

$envPrefix = "unset SSL_CERT_FILE; "
if ($ResolvedRobot) {
    $envPrefix += "export ROBOT_HOST='$ResolvedRobot'; "
}
$argStr = ($startArgs -join " ").Trim()
$bashCmd = "${envPrefix}cd '$WslRepo' && ./scripts/start_all.sh"
if ($argStr) { $bashCmd += " $argStr" }

Write-Host ""
Write-Host "Companion UI (phone LAN):  http://${LanIp}:8090" -ForegroundColor Cyan
Write-Host "Realtime WS (robot LAN):   ws://${LanIp}:8765/v1/realtime" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "[dry-run] Preflight complete. Command that would run:" -ForegroundColor Green
    Write-Host "  wsl -d $Distro -e bash -lc `"$bashCmd`""
    exit 0
}

Write-Host "[launch] Starting stack in WSL2 ($Distro)..." -ForegroundColor Cyan
& wsl -d $Distro -e bash -lc $bashCmd
exit $LASTEXITCODE
