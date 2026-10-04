# Run from this source folder using a normal Python 3.12 installation.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
py -3.12 -m venv .build-venv
& '.build-venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.build-venv\Scripts\python.exe' -m PyInstaller --noconfirm --windowed --onedir --name OpenSimConverter --collect-all mediapipe --collect-all opensim --collect-all cv2 --add-data "$PSScriptRoot\pose_landmarker_full.task;." "$PSScriptRoot\OpenSimConverter.py"
if ($LASTEXITCODE -ne 0) { throw 'EXE build failed' }
Copy-Item -LiteralPath "$PSScriptRoot\使用說明.txt" -Destination 'dist\OpenSimConverter\使用說明.txt'
Write-Output 'Completed: dist\OpenSimConverter\OpenSimConverter.exe'
