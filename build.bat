@echo off
rem Builds Lamina from this clone with a preset of CMakePresets.json: native (the default: this PC's GPUs and CPU,
rem into build\) or portable (the releases' build for other PCs, into build-portable\). lamina-server.exe and
rem lamina-engine.exe run from there (the CUDA runtime is linked in).
rem   build.bat [native or portable] [CMake options, e.g. -DLAMINA_BUILD_TESTS=ON; quote one that holds a ;]
rem It takes the MSVC the shell has set up (a developer prompt, or a portable MSVC's setup_x64.bat: cl.exe on the PATH,
rem INCLUDE and LIB), else Visual Studio's C++ tools (2022, else 2019 or 2026) and, unless the PATH has others, their
rem CMake and Ninja; the CUDA toolkit's nvcc from the PATH or CUDA_PATH (CUDACXX names another).
setlocal
cd /d "%~dp0"
set "PRESET=native"
set "ARGS=%*"
if /i "%~1"=="native" set "ARGS=%ARGS:*native=%"
if /i "%~1"=="portable" (set "PRESET=portable" & set "ARGS=%ARGS:*portable=%")
set "OUT=build"
if "%PRESET%"=="portable" set "OUT=build-portable"

where cl.exe >nul 2>nul && if defined INCLUDE if defined LIB goto have_vs
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCTOOLS=x86.x64"
set "VCVARS=vcvars64.bat"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" (set "VCTOOLS=ARM64" & set "VCVARS=vcvarsarm64.bat")
set "VS="
if not exist "%VSWHERE%" goto no_vs
rem 2022 first: nvcc before CUDA 13.3 refuses 2026's compiler
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -version [17.0^,18.0^) -requires Microsoft.VisualStudio.Component.VC.Tools.%VCTOOLS% -property installationPath`) do set "VS=%%i"
if not defined VS for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -version [16.0^,19.0^) -requires Microsoft.VisualStudio.Component.VC.Tools.%VCTOOLS% -property installationPath`) do set "VS=%%i"
if not defined VS goto no_vs
rem vcvars runs vswhere.exe from its own folder, which this variable (set by some shells) forbids
set "NoDefaultCurrentDirectoryInExePath="
call "%VS%\VC\Auxiliary\Build\%VCVARS%" >nul || exit /b 1
:have_vs

rem Windows on Arm: ggml's CPU code for Arm does not build with MSVC, so C and C++ go through LLVM's clang-cl (Visual
rem Studio's "C++ Clang Compiler for Windows" or LLVM's own; CC and CXX name others); nvcc keeps MSVC's cl
if /i not "%VSCMD_ARG_TGT_ARCH%"=="arm64" goto compilers
if defined CC goto compilers
if exist "%ProgramFiles%\LLVM\bin\clang-cl.exe" set "PATH=%ProgramFiles%\LLVM\bin;%PATH%"
if exist "%VCINSTALLDIR%Tools\Llvm\ARM64\bin\clang-cl.exe" set "PATH=%VCINSTALLDIR%Tools\Llvm\ARM64\bin;%PATH%"
set "CC=clang-cl"
set "CXX=clang-cl"
:compilers

where cmake >nul 2>nul || (echo CMake 3.22 or newer is needed: Visual Studio's "C++ CMake tools for Windows" has it, or cmake.org & exit /b 1)
where ninja >nul 2>nul || (echo Ninja is needed: Visual Studio's "C++ CMake tools for Windows" has it, or ninja-build.org & exit /b 1)
cmake --preset %PRESET% %ARGS% || exit /b 1
cmake --build --preset %PRESET% && goto done
rem CUDA's ptxas now and then fails to read a file it has just written (upstream #45): the same command gets past it
echo (the build stopped: trying it once more)
cmake --build --preset %PRESET% || exit /b 1
:done
echo Done: the programs are in %OUT%\ (%OUT%\lamina-server.exe -m MODEL.gguf runs a model)
exit /b 0

:no_vs
echo Visual Studio's C++ tools are needed: Visual Studio 2022 or its free Build Tools
echo (https://visualstudio.microsoft.com/downloads/) with the workload "Desktop development with C++";
echo or run this from a developer prompt, or after a portable MSVC's setup_x64.bat.
exit /b 1
