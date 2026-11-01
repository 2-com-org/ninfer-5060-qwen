<#
.SYNOPSIS
Installs NInfer-Qwen as its own WSL2 distribution and downloads the Qwen3.8-27B GSQ-RCO model.
.DESCRIPTION
Install.cmd in the extracted Windows release runs this. It checks the GPU and WSL, imports the
release's Linux system as the WSL distribution ninfer-5060-qwen (Ubuntu 24.04 with the server in
/opt/ninfer-5060-qwen), downloads the 10.8 GB model into it and adds Start menu entries. Running it
again resumes an interrupted download. Start-Qwen38.cmd then starts the server; Uninstall.cmd
removes everything again.
.PARAMETER InstallDir
Folder for the distribution's virtual disk, about 13 GB with the model. Default:
%LOCALAPPDATA%\ninfer-5060-qwen\wsl.
.PARAMETER ModelFile
A qwen38-27b-gsq-iq3xxs.ninfer downloaded before, copied in instead of downloading it. Its size and
SHA-256 are checked like a download's.
#>
[CmdletBinding()]
param(
    [string]$InstallDir,
    [string]$ModelFile
)

$distribution = 'ninfer-5060-qwen'
$linuxRoot = '/opt/ninfer-5060-qwen'
$modelName = 'qwen38-27b-gsq-iq3xxs.ninfer'
$modelBytes = 10846918144

function Invoke-Wsl {
    & wsl.exe --distribution $distribution --exec @args
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed (exit $LASTEXITCODE): $($args -join ' ')" }
}

function Assert-Gpu {
    # Native stderr redirected under 'Stop' would end the script at its first line.
    $ErrorActionPreference = 'Continue'
    # The server is an sm_120a build: it runs on compute capability 12.0 (RTX 50 series, RTX PRO
    # Blackwell) and needs the R580 driver for its CUDA 13 runtime.
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $smi) { throw 'No NVIDIA driver found (nvidia-smi.exe). Install the NVIDIA driver 580 or newer for your RTX 50 series card.' }
    $rows = @(& $smi.Source --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>$null)
    if ($LASTEXITCODE -ne 0 -or $rows.Count -eq 0) { throw 'nvidia-smi.exe lists no GPU. Check the NVIDIA driver.' }
    $gpus = foreach ($row in $rows) {
        $name, $capability, $driver = $row.Split(',') | ForEach-Object { $_.Trim() }
        [pscustomobject]@{ Name = $name; Capability = $capability; Driver = $driver }
    }
    $usable = @($gpus | Where-Object Capability -eq '12.0')
    if ($usable.Count -eq 0) {
        throw "This release runs on RTX 50 series GPUs (compute capability 12.0); found $(($gpus | ForEach-Object { "$($_.Name) ($($_.Capability))" }) -join ', ')."
    }
    $major = [int]($usable[0].Driver.Split('.')[0])
    if ($major -lt 580) { throw "The NVIDIA driver is $($usable[0].Driver); this release needs 580 or newer. Update it and run Install.cmd again." }
    Write-Host "GPU: $($usable[0].Name), driver $($usable[0].Driver)"
}

function Assert-Wsl {
    $ErrorActionPreference = 'Continue'
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { $code = 1 } else {
        & wsl.exe --version *> $null
        $code = $LASTEXITCODE
    }
    if ($code -ne 0) {
        throw 'WSL 2 is missing or too old. Open a terminal as administrator, run "wsl --install --no-distribution" (or "wsl --update" if WSL is installed), restart Windows if asked, then run Install.cmd again.'
    }
}

function Get-InstalledVersion {
    $ErrorActionPreference = 'Continue'
    $names = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { $_.Trim() })
    if ($names -notcontains $distribution) { return $null }
    $version = & wsl.exe --distribution $distribution --exec /bin/cat "$linuxRoot/VERSION" 2>$null
    if ($LASTEXITCODE -ne 0) { return '' }
    return "$version".Trim()
}

function Start-WslProcess([string[]]$Arguments, [switch]$Reads, [switch]$Writes) {
    # CreateProcess quoting, including embedded quotes and trailing backslashes.
    $line = ($Arguments | ForEach-Object {
        if ($_ -and $_ -notmatch '[\s"]') { $_ } else {
            $escaped = [regex]::Replace($_, '(\\*)"', '$1$1\"')
            '"' + [regex]::Replace($escaped, '(\\+)$', '$1$1') + '"'
        }
    }) -join ' '
    $start = New-Object Diagnostics.ProcessStartInfo 'wsl.exe', $line
    $start.UseShellExecute = $false
    $start.RedirectStandardInput = [bool]$Writes
    $start.RedirectStandardOutput = [bool]$Reads
    # Process.Start writes the console input encoding's byte order mark into a redirected stdin;
    # under a UTF-8 code page that would put three bytes before the copied file.
    $encoding = [Console]::InputEncoding
    try {
        if ($Writes) { try { [Console]::InputEncoding = New-Object Text.UTF8Encoding $false } catch { } }
        return [Diagnostics.Process]::Start($start)
    } finally {
        if ($Writes) { try { [Console]::InputEncoding = $encoding } catch { } }
    }
}

function Copy-ModelFile([string]$Source, [string]$Target) {
    # The bytes go into WSL through a pipe: writing the file through \\wsl$ runs at a few MB/s, a
    # pipe at hundreds. A file in another WSL distribution is read out the same way.
    $length = (Get-Item -LiteralPath $Source).Length
    if ($length -ne $modelBytes) { throw "$Source is $length bytes; $modelName is $modelBytes." }
    $sourceProcess = $null
    if ($Source -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)(\\.*)$') {
        $sourceProcess = Start-WslProcess @('--distribution', $Matches[1], '--exec', '/bin/cat', '--', $Matches[2].Replace('\', '/')) -Reads
        $reader = $sourceProcess.StandardOutput.BaseStream
    } else {
        $reader = [IO.File]::OpenRead($Source)
    }
    $targetProcess = Start-WslProcess @('--distribution', $distribution, '--exec', '/bin/sh', '-c', 'cat > "$1"', 'sh', $Target) -Writes
    try {
        # Written and closed on the base stream: the text writer around it can add a byte order mark.
        $writer = $targetProcess.StandardInput.BaseStream
        $buffer = New-Object byte[] (16MB)
        $copied = 0L
        $reported = 0
        while (($read = $reader.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $writer.Write($buffer, 0, $read)
            $copied += $read
            $percent = [int](100 * $copied / $modelBytes)
            if ($percent -ge $reported + 10) { Write-Host "  $percent%"; $reported = $percent }
        }
        $writer.Close()
        $targetProcess.WaitForExit()
        if ($targetProcess.ExitCode -ne 0) { throw "Writing $Target in WSL failed (exit $($targetProcess.ExitCode))." }
        if ($sourceProcess) {
            $sourceProcess.WaitForExit()
            if ($sourceProcess.ExitCode -ne 0) { throw "Reading $Source failed (exit $($sourceProcess.ExitCode))." }
        }
        if ($copied -ne $modelBytes) { throw "Copied $copied bytes of $Source; expected $modelBytes." }
        $written = & wsl.exe --distribution $distribution --exec /usr/bin/stat -c %s $Target
        if ("$written".Trim() -ne "$modelBytes") { throw "$Target in WSL is $written bytes after the copy; expected $modelBytes." }
    } finally {
        $reader.Dispose()
        if (-not $targetProcess.HasExited) { $targetProcess.Kill() }
        $targetProcess.Dispose()
        if ($sourceProcess) {
            if (-not $sourceProcess.HasExited) { $sourceProcess.Kill() }
            $sourceProcess.Dispose()
        }
    }
}

function Add-Shortcut([string]$Path, [string]$Target, [string]$Arguments, [string]$WorkingDirectory, [string]$Description) {
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($Path)
    $link.TargetPath = $Target
    $link.Arguments = $Arguments
    $link.WorkingDirectory = $WorkingDirectory
    $link.Description = $Description
    $link.Save()
}

function Install-NInferQwen {
    $package = Split-Path -Parent $PSScriptRoot
    $versionFile = Join-Path $package 'VERSION'
    if (-not (Test-Path -LiteralPath $versionFile)) { throw "Missing $versionFile. Extract the whole zip (right-click it, Extract All) and run Install.cmd from the extracted folder." }
    $version = (Get-Content -LiteralPath $versionFile -Raw).Trim()
    $rootfs = Join-Path $package "ninfer-5060-qwen-$version-wsl.tar.gz"
    if (-not (Test-Path -LiteralPath $rootfs)) { throw "Missing $rootfs. Extract the whole zip (right-click it, Extract All) and run Install.cmd from the extracted folder." }
    if ($ModelFile -and -not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) { throw "-ModelFile $ModelFile does not exist." }
    $appDirectory = Join-Path $env:LOCALAPPDATA 'ninfer-5060-qwen'
    if (-not $InstallDir) { $InstallDir = Join-Path $appDirectory 'wsl' }
    $InstallDir = [IO.Path]::GetFullPath($InstallDir)

    Assert-Gpu
    Assert-Wsl
    $installed = Get-InstalledVersion
    if ($null -eq $installed) {
        if (Test-Path -LiteralPath (Join-Path $InstallDir 'ext4.vhdx')) { throw "$InstallDir already holds a WSL disk (ext4.vhdx). Choose another -InstallDir." }
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        # The system takes about 1.7 GB, the model 10.8 GB; WSL's disk grows as they are written.
        $free = (New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($InstallDir))).AvailableFreeSpace
        if ($free -lt 14GB) { throw "$InstallDir has $([math]::Round($free / 1GB, 1)) GB free; NInfer-Qwen needs about 14 GB. Pass -InstallDir on another drive." }
        Write-Host "Installing NInfer-Qwen $version as the WSL distribution $distribution in $InstallDir"
        & wsl.exe --import $distribution $InstallDir $rootfs --version 2
        if ($LASTEXITCODE -ne 0) { throw "wsl --import failed (exit $LASTEXITCODE)." }
        Write-Host 'Checking the installed files'
        Invoke-Wsl /bin/sh -c "cd '$linuxRoot' && sha256sum --quiet --check SHA256SUMS.txt"
    } elseif (-not $installed) {
        throw "The WSL distribution $distribution exists but holds no NInfer-Qwen. Run Uninstall.cmd, then run this again."
    } elseif ($installed -ne $version) {
        throw "NInfer-Qwen $installed is installed. Run its Uninstall.cmd (it also removes the downloaded model), then run this again."
    } else {
        Write-Host "NInfer-Qwen $version is installed; checking the model"
    }

    Invoke-Wsl /bin/mkdir -p "$linuxRoot/models"
    if ($ModelFile) {
        # Copied in as the download's partial file: download-model.sh checks its size and SHA-256,
        # then moves it into place.
        $target = "$linuxRoot/models/$modelName"
        & wsl.exe --distribution $distribution --exec /bin/test -f $target
        if ($LASTEXITCODE -eq 0) { Write-Host "$modelName is already installed; -ModelFile is not copied." } else {
            Write-Host "Copying $ModelFile into WSL"
            Copy-ModelFile ([IO.Path]::GetFullPath($ModelFile)) "$target.part"
        }
    }
    Invoke-Wsl /bin/bash "$linuxRoot/download-model.sh"

    # The Windows side: the starter and uninstaller in %LOCALAPPDATA%, so this folder can be deleted,
    # and Start menu entries for them.
    New-Item -ItemType Directory -Path (Join-Path $appDirectory 'scripts') -Force | Out-Null
    foreach ($file in @('Start-Qwen38.cmd', 'scripts\Uninstall.ps1')) {
        Copy-Item -LiteralPath (Join-Path $package $file) -Destination (Join-Path $appDirectory $file) -Force
    }
    $menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'NInfer-Qwen'
    New-Item -ItemType Directory -Path $menu -Force | Out-Null
    Add-Shortcut (Join-Path $menu 'Qwen3.8-27B.lnk') (Join-Path $appDirectory 'Start-Qwen38.cmd') '' $appDirectory 'Start the Qwen3.8-27B server and open its WebUI'
    # The uninstaller removes its own folder, so this entry runs the script directly: a batch file
    # that is deleted while it runs ends with "The batch file cannot be found".
    $powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    Add-Shortcut (Join-Path $menu 'Uninstall NInfer-Qwen.lnk') $powershell `
        "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $appDirectory 'scripts\Uninstall.ps1')`"" `
        $env:USERPROFILE 'Remove NInfer-Qwen and its model'
    Write-Host ''
    Write-Host 'NInfer-Qwen is installed. Start it from the Start menu (NInfer-Qwen > Qwen3.8-27B) or with Start-Qwen38.cmd.'
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    # WSL prints its own messages as UTF-16 unless asked for UTF-8.
    $env:WSL_UTF8 = '1'
    try { Install-NInferQwen } catch {
        Write-Error $_ -ErrorAction Continue
        exit 1
    }
}
