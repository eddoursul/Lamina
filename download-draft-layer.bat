@echo off
rem Lamina's draft layer for Qwen3.8-Flash-Next, one file of 0.8 GB, from
rem https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP at a pinned revision, checked by its SHA-256.
rem
rem   download-draft-layer.bat FOLDER
rem
rem lamina-server takes the file from beside the model or beside the program; elsewhere name it in the config's
rem "args": "--drafter", "FOLDER\Lamina-Qwen3.8-Flash-Next-MTP.gguf". A download that broke off goes on from where it
rem stopped. HF_ENDPOINT names a mirror. It needs curl.exe and certutil.exe, which Windows 10 (1803) and newer have.
setlocal
set "NAME=Lamina-Qwen3.8-Flash-Next-MTP.gguf"
if "%~1"=="" goto usage
if not "%~2"=="" goto usage
set "DEST=%~f1"
set "REV=e0d95e4236a43f3e02153c290211b74ba17a6f90"
set "SHA=eeaa94d4cd1e51f5b548aacbb72f8a0f6989cc11c62923b5cece11b33712c7db"
set "BASE=https://huggingface.co"
if defined HF_ENDPOINT set "BASE=%HF_ENDPOINT%"
set "URL=%BASE%/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP/resolve/%REV%/%NAME%"
where curl >nul 2>nul
if errorlevel 1 goto nocurl
if not exist "%DEST%\" mkdir "%DEST%"
if not exist "%DEST%\" goto nofolder
set "FILE=%DEST%\%NAME%"
if not exist "%FILE%" goto part
call :sha "%FILE%"
if /i "%HASH%"=="%SHA%" goto there
echo "%FILE%" is not the file expected (its SHA-256 differs): move it away and run this again
exit /b 1
:there
echo %NAME%: there already, checked
goto done
:part
if not exist "%FILE%.part" goto fetch
call :sha "%FILE%.part"
if /i "%HASH%"=="%SHA%" goto keep
:fetch
echo %NAME% ...
curl -L --fail --retry 5 --retry-delay 5 -C - -o "%FILE%.part" "%URL%"
if errorlevel 1 goto failed
call :sha "%FILE%.part"
if /i "%HASH%"=="%SHA%" goto keep
echo the download is not the file expected (its SHA-256 differs): delete "%FILE%.part" and run this again
exit /b 1
:keep
move /y "%FILE%.part" "%FILE%" >nul
if errorlevel 1 exit /b 1
echo %NAME%: downloaded, checked
:done
echo.
echo The draft layer is "%FILE%"
echo lamina-server takes it from beside the model or the program; elsewhere name it in the config's "args":
echo   "--drafter", "%FILE:\=/%"
exit /b 0

:usage
echo usage: %~nx0 FOLDER
echo Downloads Lamina's draft layer for Qwen3.8-Flash-Next (%NAME%, 0.8 GB) into FOLDER: the model's
echo folder, where lamina-server finds it
exit /b 2

:nocurl
echo curl.exe was not found: Windows 10 (1803) and newer have it
exit /b 1

:nofolder
echo the folder "%DEST%" could not be made
exit /b 1

:failed
echo the download failed. Run this again to go on from where it stopped; if it fails the same way, delete
echo "%FILE%.part" first
exit /b 1

:sha
rem HASH: the SHA-256 of %1 (the second line certutil prints; older Windows put spaces between its bytes)
set "HASH="
for /f "skip=1 delims=" %%h in ('certutil -hashfile %1 SHA256') do if not defined HASH set "HASH=%%h"
if not defined HASH exit /b 1
set "HASH=%HASH: =%"
exit /b 0
