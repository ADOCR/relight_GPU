param(
    [Parameter(Mandatory=$true)][string]$QtRoot,
    [Parameter(Mandatory=$true)][string]$VcpkgRoot,
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$BuildDirectory = '',
    [ValidateRange(1,64)][int]$Jobs = 4,
    [string]$MsvcDepsPrefix = '',
    [switch]$CpuOnly
)
$ErrorActionPreference='Stop'
$projectRootRTI=Split-Path -Parent $PSScriptRoot
if(-not $BuildDirectory){$BuildDirectory=Join-Path $projectRootRTI 'build'}
$vswhereRTI=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if(-not (Test-Path -LiteralPath $vswhereRTI)){throw 'Visual Studio with C++ tools is required.'}
$visualStudioRTI=& $vswhereRTI -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(-not $visualStudioRTI){throw 'Visual Studio C++ tools were not found.'}
& (Join-Path $visualStudioRTI 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
$cmakeCommandRTI=Get-Command cmake -ErrorAction SilentlyContinue
$cmakeExeRTI=if($cmakeCommandRTI){$cmakeCommandRTI.Source}else{Join-Path $visualStudioRTI 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'}
$ninjaRTI=Join-Path $visualStudioRTI 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$argumentsRTI=@('-S',$projectRootRTI,'-B',$BuildDirectory,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',"-DCMAKE_PREFIX_PATH=$QtRoot","-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot/scripts/buildsystems/vcpkg.cmake","-DCMAKE_MAKE_PROGRAM=$ninjaRTI",'-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /W3 /GR /EHsc /utf-8')
if($CudaRoot){$argumentsRTI+="-DCUDAToolkit_ROOT=$CudaRoot"}
if($CpuOnly){$argumentsRTI+='-DRELIGHT_ENABLE_CUDA=OFF'}else{$argumentsRTI+='-DRELIGHT_ENABLE_CUDA=ON'}
if($MsvcDepsPrefix){$argumentsRTI+="-DRELIGHT_MSVC_DEPS_PREFIX=$MsvcDepsPrefix"}
& $cmakeExeRTI @argumentsRTI
if($LASTEXITCODE -ne 0){throw 'CMake configuration failed.'}
& $cmakeExeRTI --build $BuildDirectory --target relightlab relight-cli --parallel $Jobs
if($LASTEXITCODE -ne 0){throw 'Build failed.'}
Write-Host "Build ready: $BuildDirectory"
