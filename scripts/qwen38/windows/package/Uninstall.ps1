<#
.SYNOPSIS
Removes NInfer-Qwen: its WSL distribution with the server and the model, and its Windows files.
.DESCRIPTION
Uninstall.cmd and the Start menu entry run this. It unregisters the WSL distribution ninfer-5060-qwen, which deletes its
virtual disk with the server and the downloaded model, and removes the Start menu entries and the
files Install.cmd copied to %LOCALAPPDATA%\ninfer-5060-qwen. The server logs there stay. A running
server stops. Other WSL distributions are not touched.
.PARAMETER Yes
Remove without asking.
#>
[CmdletBinding()]
param([switch]$Yes)

$distribution = 'ninfer-5060-qwen'

function Get-DistributionFolder {
    foreach ($key in Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' -ErrorAction SilentlyContinue) {
        $entry = Get-ItemProperty -LiteralPath $key.PSPath
        if ($entry.DistributionName -eq $distribution) { return ($entry.BasePath -replace '^\\\\\?\\', '') }
    }
    return $null
}

function Uninstall-NInferQwen {
    $appDirectory = Join-Path $env:LOCALAPPDATA 'ninfer-5060-qwen'
    $menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'NInfer-Qwen'
    $folder = Get-DistributionFolder
    # The logs folder alone is not an installation: the other launchers write there too.
    $copied = @('Start-Qwen38.cmd', 'scripts') | Where-Object { Test-Path -LiteralPath (Join-Path $appDirectory $_) }
    if (-not $folder -and -not $copied -and -not (Test-Path -LiteralPath $menu)) {
        Write-Host 'NInfer-Qwen is not installed.'
        return
    }
    if (-not $Yes) {
        Write-Host "This removes the WSL distribution $distribution with the server and the downloaded model,"
        Write-Host "the Start menu entries, and the launcher files in $appDirectory."
        if ((Read-Host 'Type yes to remove NInfer-Qwen') -ne 'yes') { Write-Host 'Nothing was removed.'; return }
    }
    if ($folder) {
        & wsl.exe --unregister $distribution
        if ($LASTEXITCODE -ne 0) { throw "wsl --unregister $distribution failed (exit $LASTEXITCODE)." }
        # Unregistering deletes the virtual disk; the folder Install.cmd made for it stays.
        if ((Test-Path -LiteralPath $folder) -and -not (Get-ChildItem -LiteralPath $folder -Force)) { Remove-Item -LiteralPath $folder }
    }
    if (Test-Path -LiteralPath $menu) { Remove-Item -LiteralPath $menu -Recurse -Force }
    # Only the files Install.cmd copied: the launchers of a Linux release or checkout keep their logs
    # in the same folder.
    Set-Location -LiteralPath $env:TEMP
    foreach ($file in @('Start-Qwen38.cmd', 'scripts\Uninstall.ps1', 'scripts')) {
        $path = Join-Path $appDirectory $file
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
    if ((Test-Path -LiteralPath $appDirectory) -and -not (Get-ChildItem -LiteralPath $appDirectory -Force)) { Remove-Item -LiteralPath $appDirectory }
    Write-Host 'NInfer-Qwen is removed.'
    if (Test-Path -LiteralPath (Join-Path $appDirectory 'logs')) { Write-Host "The server logs stay in $(Join-Path $appDirectory 'logs')." }
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    $env:WSL_UTF8 = '1'
    $code = 0
    try { Uninstall-NInferQwen } catch {
        Write-Error $_ -ErrorAction Continue
        $code = 1
    }
    # The Start menu entry runs this script without a batch file, so the pause is here.
    if (-not $Yes) { Read-Host 'Press Enter to close' | Out-Null }
    exit $code
}
