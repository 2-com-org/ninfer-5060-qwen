<#
.SYNOPSIS
Runs Qwen3.8-27B GSQ-RCO IQ3_XXS through WSL2 and opens its WebUI.
.DESCRIPTION
Keep this terminal open while using the server; it shows the server log, which is also saved under
%LOCALAPPDATA%\ninfer-5060-qwen\logs. Ctrl+C or closing the window stops the server and releases GPU
memory. The server and model live in WSL: in the release archive that holds this script's windows\
folder (download-model.sh fetches the model), or in the checkout that holds it (scripts/qwen38/setup.sh
builds and prepares them). The launch settings come from serve.sh, which documents each of them.
Parameters left out take serve.sh's defaults.
.PARAMETER Context
Tokens per request (serve.sh --context). Default 262144 with up to 3 requests.
.PARAMETER Concurrency
Requests served at once, 1 to 32 (serve.sh --concurrency).
.PARAMETER Spec
mtp (default) or none.
.PARAMETER DraftTokens
MTP drafts per round (default 4 with one request, 3 with more).
.PARAMETER CopyTokens
Tokens copied per round from text the reply repeats; 0 disables.
.PARAMETER PrefixCache
catalog (default) or alt.
.PARAMETER LeaseGrowth
With -Concurrency 2 or more: grow each request's reservation as it generates.
.PARAMETER RamCacheGiB
Pinned system RAM for finished conversations, 2 to 40 GiB.
.PARAMETER Temperature
Default sampling temperature, 0 (greedy) to 2.
.PARAMETER Thinking
Think by default.
.PARAMETER Port
Server port, default 8080.
.PARAMETER LocalOnly
Accept connections from this PC only.
.PARAMETER Model
Linux path of the model inside WSL (default: models/qwen38-27b-gsq-iq3xxs.ninfer in the release or
checkout).
.PARAMETER Server
Linux path of ninfer-serve inside WSL (default: bin/ninfer-serve in the release,
build-qwen38/apps/ninfer-serve in a checkout).
.PARAMETER NoBrowser
Do not open the WebUI.
.PARAMETER Dashboard
Also run the monitor dashboard (tools/monitor/monitor.py, checkout only) at http://localhost:8090:
throughput, acceptance, time to first token, KV and cache pressure, GPU and host load. The server's
own WebUI has a lighter monitor at /monitor/ without it.
.PARAMETER DashboardPort
Dashboard port, default 8090. Its stats listener takes -Port + 1.
.PARAMETER Python
Python inside WSL that runs -Dashboard (standard library only), default python3.
.PARAMETER Distribution
WSL distribution. Default: the one that holds this release or checkout, else WSL's default
distribution.
.EXAMPLE
.\Start-Qwen38.ps1 -Context 65536 -LocalOnly -NoBrowser
.EXAMPLE
.\Start-Qwen38.ps1 -Concurrency 4
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 262144)][int]$Context,
    [ValidateRange(1, 32)][int]$Concurrency,
    [ValidateSet('mtp', 'none')][string]$Spec,
    [ValidateRange(1, 15)][int]$DraftTokens,
    [ValidateRange(0, 63)][int]$CopyTokens,
    [ValidateSet('catalog', 'alt')][string]$PrefixCache,
    [switch]$LeaseGrowth,
    [ValidateRange(2, 40)][int]$RamCacheGiB,
    [ValidateRange(0.0, 2.0)][ValidateScript({ -not [double]::IsNaN($_) })][double]$Temperature,
    [switch]$Thinking,
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [switch]$LocalOnly,
    [string]$Model,
    [string]$Server,
    [switch]$NoBrowser,
    [switch]$Dashboard,
    [ValidateRange(1, 65535)][int]$DashboardPort = 8090,
    [string]$Python = 'python3',
    [string]$Distribution,
    [ValidateRange(1, 3600)][int]$StartupTimeoutSeconds = 600
)

function ConvertTo-NInferWslPath([string]$Path, [string]$Distribution) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if ($absolute -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)(\\.*)$') {
        if ($Distribution -and $Matches[1] -ne $Distribution) {
            throw "WSL path belongs to distribution '$($Matches[1])'; selected distribution is '$Distribution'."
        }
        return $Matches[2].Replace('\', '/')
    }
    if ($absolute -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Use a local Windows drive or WSL share path for WSL launch: $Path"
    }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
}

function ConvertTo-NInferWindowsArguments([string[]]$Values) {
    # Quote the CreateProcess argument vector, including embedded quotes and trailing slashes.
    return (($Values | ForEach-Object {
        if ($_ -and $_ -notmatch '[\s"]') { $_ } else {
            $escaped = [regex]::Replace($_, '(\\*)"', '$1$1\"')
            $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
            '"' + $escaped + '"'
        }
    }) -join ' ')
}

function New-NInferProcessJob {
    # The job owns the Windows WSL client. A separate stdin lifetime pipe makes the Linux
    # supervisor stop its server when this Windows owner or its client disappears.
    if (-not ('NInferLauncher.ProcessJob' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace NInferLauncher {
    public sealed class ProcessJob : IDisposable {
        private SafeFileHandle handle;
        private FileStream stdout, stderr;
        private Task stdoutCopy, stderrCopy;
        private readonly object consoleLock = new object();

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimits {
            public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }
        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimits {
            public BasicLimits Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(SafeFileHandle job, int infoClass,
                                                           ref ExtendedLimits info, uint size);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(SafeFileHandle job, IntPtr process);

        public ProcessJob() {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle.IsInvalid) {
                int error = Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new Win32Exception(error, "Cannot create the NInfer process ownership job.");
            }
            var limits = new ExtendedLimits();
            limits.Basic.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(handle, 9, ref limits,
                                         (uint)Marshal.SizeOf(typeof(ExtendedLimits)))) {
                int error = Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new Win32Exception(error, "Cannot configure the NInfer process ownership job.");
            }
        }
        public void Assign(Process process) {
            if (!AssignProcessToJobObject(handle, process.Handle)) {
                throw new Win32Exception(Marshal.GetLastWin32Error(),
                                         "Cannot assign WSL to the NInfer process ownership job.");
            }
        }
        public Process Start(string executable, string arguments, string output, string error) {
            stdout = new FileStream(output, FileMode.Create, FileAccess.Write, FileShare.Read, 1, true);
            stderr = new FileStream(error, FileMode.Create, FileAccess.Write, FileShare.Read, 1, true);
            var process = new Process();
            process.StartInfo = new ProcessStartInfo(executable, arguments) {
                UseShellExecute = false, CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
                RedirectStandardInput = true, RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            bool started = false;
            try {
                process.Start();
                started = true;
                Assign(process);
                stdoutCopy = Mirror(process.StandardOutput.BaseStream, stdout);
                stderrCopy = Mirror(process.StandardError.BaseStream, stderr);
                process.StandardInput.NewLine = "\n";
                process.StandardInput.WriteLine("start");
                process.StandardInput.Flush();
                return process;
            } catch {
                // No handshake was sent if assignment failed, so Linux cannot launch a server.
                if (started && !process.HasExited) { process.Kill(); process.WaitForExit(); }
                process.Dispose();
                throw;
            }
        }
        // Writes a server stream to its log file and to this console as it arrives, so the
        // terminal shows the server log live.
        private async Task Mirror(Stream source, FileStream log) {
            var buffer = new byte[8192];
            var decoder = new UTF8Encoding(false).GetDecoder();
            var text = new char[Encoding.UTF8.GetMaxCharCount(buffer.Length)];
            int read;
            while ((read = await source.ReadAsync(buffer, 0, buffer.Length).ConfigureAwait(false)) > 0) {
                await log.WriteAsync(buffer, 0, read).ConfigureAwait(false);
                int count = decoder.GetChars(buffer, 0, read, text, 0);
                lock (consoleLock) {
                    Console.Out.Write(text, 0, count);
                    Console.Out.Flush();
                }
            }
        }
        public void FlushLogs() {
            if (stdoutCopy != null) stdoutCopy.Wait();
            if (stderrCopy != null) stderrCopy.Wait();
            if (stdout != null) stdout.Flush();
            if (stderr != null) stderr.Flush();
        }
        public void Dispose() {
            handle.Dispose();
            try { FlushLogs(); } finally {
                if (stdout != null) stdout.Dispose();
                if (stderr != null) stderr.Dispose();
            }
        }
    }
}
'@
    }
    return New-Object NInferLauncher.ProcessJob
}

function Get-NInferNetworkUrls([int]$Port) {
    # Addresses other devices can reach; WSL's own virtual adapters and link-local addresses are skipped.
    $addresses = foreach ($adapter in [Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()) {
        if ($adapter.OperationalStatus -ne 'Up' -or $adapter.NetworkInterfaceType -in @('Loopback', 'Tunnel') -or $adapter.Name -like 'vEthernet*') { continue }
        foreach ($unicast in $adapter.GetIPProperties().UnicastAddresses) {
            $address = $unicast.Address
            if ($address.AddressFamily -eq 'InterNetwork' -and -not $address.ToString().StartsWith('169.254.')) { $address.ToString() }
        }
    }
    @($addresses | Sort-Object -Unique | ForEach-Object { "http://${_}:$Port" })
}

function Test-NInferPortInUse([int]$Port) {
    $connection = New-Object Net.Sockets.TcpClient
    try {
        $pending = $connection.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($pending.AsyncWaitHandle.WaitOne(500)) { try { $connection.EndConnect($pending) } catch { } }
        return $connection.Connected
    } finally { $connection.Dispose() }
}

function Get-QwenServeArguments([hashtable]$Bound) {
    # serve.sh owns the launch settings; forward only what was given so its defaults apply.
    $names = [ordered]@{
        Context = '--context'; Concurrency = '--concurrency'; Spec = '--spec'; DraftTokens = '--draft-tokens'
        CopyTokens = '--copy-tokens'; PrefixCache = '--prefix-cache'; RamCacheGiB = '--ram-cache-gib'
        Port = '--port'; Model = '--model'; Server = '--server'
    }
    $arguments = @()
    foreach ($name in $names.Keys) { if ($Bound.ContainsKey($name)) { $arguments += @($names[$name], "$($Bound[$name])") } }
    if ($Bound.ContainsKey('Temperature')) {
        $arguments += @('--temperature', ([double]$Bound['Temperature']).ToString('0.###', [Globalization.CultureInfo]::InvariantCulture))
    }
    if ($Bound['LeaseGrowth']) { $arguments += '--lease-growth' }
    if ($Bound['Thinking']) { $arguments += '--thinking' }
    if ($Bound['LocalOnly']) { $arguments += '--local-only' }
    return $arguments
}

function Invoke-QwenLauncher {
    param([hashtable]$Bound, [int]$Port, [switch]$LocalOnly, [switch]$NoBrowser, [int]$DashboardPort,
          [string]$Python, [string]$Distribution, [int]$StartupTimeoutSeconds)
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) { throw 'WSL is missing: install it with "wsl --install -d Ubuntu", then unpack the release (or set up a checkout) inside it.' }
    # A release archive keeps bin\ and serve.sh beside this windows\ folder; a checkout keeps serve.sh
    # in scripts\qwen38, next to windows\ but without bin\.
    $release = Test-Path -LiteralPath (Join-Path $PSScriptRoot '..\bin\ninfer-serve')
    $base = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $(if ($release) { '..' } else { '..\..\..' })))
    if (-not $Distribution -and $base -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)\\') { $Distribution = $Matches[1] }
    $distributionArguments = if ($Distribution) { @('--distribution', $Distribution) } else { @() }
    $root = ConvertTo-NInferWslPath $base $Distribution
    if ($release) {
        $serve = "$root/serve.sh"
        $supervisor = "$root/windows/wsl-supervisor.sh"
        $defaultServer = "$root/bin/ninfer-serve"
        $prepare = "Run ./download-model.sh in $root"
        if ($DashboardPort) { throw '-Dashboard runs tools/monitor from a checkout; the release has the WebUI monitor at /monitor/ instead.' }
    } else {
        $serve = "$root/scripts/qwen38/serve.sh"
        $supervisor = "$root/scripts/qwen38/windows/wsl-supervisor.sh"
        $defaultServer = "$root/build-qwen38/apps/ninfer-serve"
        $prepare = "Run scripts/qwen38/setup.sh in $root"
    }

    # Resolve the settings first: serve.sh validates them and checks the server and model exist.
    $serveArguments = Get-QwenServeArguments $Bound
    $plan = & $wsl.Source @distributionArguments --exec /bin/bash $serve @serveArguments --print 2>&1
    if ($LASTEXITCODE -ne 0) { throw ($plan -join [Environment]::NewLine) }
    & $wsl.Source @distributionArguments --exec /bin/bash -c 'test -x "$1" && test -f "$2"' check `
        $(if ($Bound.ContainsKey('Server')) { $Bound['Server'] } else { $defaultServer }) `
        $(if ($Bound.ContainsKey('Model')) { $Bound['Model'] } else { "$root/models/qwen38-27b-gsq-iq3xxs.ninfer" })
    if ($LASTEXITCODE -ne 0) { throw "The server or model is missing in WSL. $prepare first, or pass -Server and -Model." }

    if (Test-NInferPortInUse $Port) { throw "Port $Port is already in use. Stop the existing server or choose -Port." }
    $statsPort = 0
    if ($DashboardPort) {
        $statsPort = $Port + 1
        if ($Port -eq 65535) { throw '-Dashboard needs -Port below 65535 (its stats listener takes the next port).' }
        if ($DashboardPort -in @($Port, $statsPort)) { throw "-DashboardPort $DashboardPort collides with the server's ports $Port and $statsPort; choose another." }
        foreach ($taken in @($DashboardPort, $statsPort)) { if (Test-NInferPortInUse $taken) { throw "Port $taken is already in use; choose another -Port or -DashboardPort." } }
    }

    $logDirectory = Join-Path $env:LOCALAPPDATA ('ninfer-5060-qwen\logs\' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $pidFile = Join-Path $logDirectory 'server.pid'
    $stdout = Join-Path $logDirectory 'server.stdout.log'
    $stderr = Join-Path $logDirectory 'server.stderr.log'
    $command = @('/bin/bash', $serve) + $serveArguments
    $sidecar = @()
    if ($DashboardPort) {
        $requestLog = ConvertTo-NInferWslPath (Join-Path $logDirectory 'requests.jsonl') $Distribution
        $command += @('--', '--stats-port', "$statsPort", '--request-log-jsonl', $requestLog)
        $monitor = @($Python, "$root/tools/monitor/monitor.py", '--bind', $(if ($LocalOnly) { '127.0.0.1' } else { '0.0.0.0' }),
            '--port', "$DashboardPort", '--stats-url', "http://127.0.0.1:$statsPort", '--jsonl', $requestLog)
        $sidecar = @('--sidecar', "$($monitor.Count)") + $monitor
    }
    $arguments = $distributionArguments + @('--exec', '/usr/bin/setsid', '--wait', '/bin/bash', $supervisor) + $sidecar +
        @((ConvertTo-NInferWslPath $pidFile $Distribution)) + $command
    $url = "http://localhost:$Port"
    $process = $null
    $job = $null
    try {
        $job = New-NInferProcessJob
        Write-Host "Starting Qwen3.8-27B at $url. The server log follows and is saved in $logDirectory"
        Write-Host 'Keep this terminal open. Press Ctrl+C to stop the server and release GPU memory.'
        $process = $job.Start($wsl.Source, (ConvertTo-NInferWindowsArguments $arguments), $stdout, $stderr)
        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            if ($process.HasExited) {
                $job.FlushLogs()
                $details = if (Test-Path -LiteralPath $stderr) { (Get-Content -LiteralPath $stderr -Tail 25) -join [Environment]::NewLine } else { '' }
                throw "NInfer exited during startup (code $($process.ExitCode)). $details"
            }
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction Stop
                if ($response.StatusCode -eq 200) { $ready = $true; break }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) { throw "NInfer did not become ready within $StartupTimeoutSeconds seconds. Inspect $stderr and $stdout." }
        Write-Host "Qwen3.8-27B is ready: WebUI $url  Monitor $url/monitor/  OpenAI base URL $url/v1"
        if ($DashboardPort) { Write-Host "Dashboard: http://localhost:$DashboardPort" }
        if (-not $LocalOnly) {
            $networkUrls = @(Get-NInferNetworkUrls $Port)
            if ($DashboardPort) { $networkUrls += @(Get-NInferNetworkUrls $DashboardPort) }
            if ($networkUrls.Count -gt 0) { Write-Host "Other devices on this network: $($networkUrls -join ', ')" }
        }
        if (-not $NoBrowser) { Start-Process $url | Out-Null }
        while (-not $process.WaitForExit(500)) { }
        $job.FlushLogs()
        if ($process.ExitCode -ne 0) { throw "NInfer exited with code $($process.ExitCode). Inspect $stderr." }
    } finally {
        if ($null -ne $process -and -not $process.HasExited) {
            $process.StandardInput.Close()
            if (Test-Path -LiteralPath $pidFile) {
                $linuxPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
                if ($linuxPid -match '^[1-9][0-9]*$') {
                    try { & $wsl.Source @distributionArguments --exec /bin/kill -TERM -- "-$linuxPid" 2>&1 | Out-Null } catch { }
                    if (-not $process.WaitForExit(10000)) {
                        try { & $wsl.Source @distributionArguments --exec /bin/kill -KILL -- "-$linuxPid" 2>&1 | Out-Null } catch { }
                    }
                }
            }
            if (-not $process.WaitForExit(3000)) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
        }
        if ($null -ne $job) { $job.Dispose() }
        if ($null -ne $process) { $process.Dispose() }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    try {
        Invoke-QwenLauncher -Bound $PSBoundParameters -Port $Port -LocalOnly:$LocalOnly -NoBrowser:$NoBrowser `
            -DashboardPort $(if ($Dashboard) { $DashboardPort } else { 0 }) -Python $Python -Distribution $Distribution `
            -StartupTimeoutSeconds $StartupTimeoutSeconds
    } catch {
        Write-Error $_ -ErrorAction Continue
        exit 1
    }
}
