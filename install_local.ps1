Write-Host "?? STARTING PORTABLE SETUP..." -ForegroundColor Cyan
$BasePath = Get-Location
$PythonDir = "$BasePath\python_bin"
$ZipFile = "$BasePath\python_zip.zip"
$PythonUrl = "https://www.python.org/ftp/python/3.10.5/python-3.10.5-embed-amd64.zip"

# --- DOWNLOAD PYTHON ---
if (-Not (Test-Path $PythonDir)) {
    Write-Host "?? Downloading Python..."
    Invoke-WebRequest -Uri $PythonUrl -OutFile $ZipFile
    Expand-Archive -Path $ZipFile -DestinationPath $PythonDir -Force
    Remove-Item $ZipFile
}

# --- FIX PIP SUPPORT ---
$Pth = "$PythonDir\python310._pth"
if (Test-Path $Pth) {
    (Get-Content $Pth).Replace("#import site", "import site") | Set-Content $Pth
}

# --- INSTALL PIP ---
if (-Not (Test-Path "$PythonDir\Scripts\pip.exe")) {
    Write-Host "?? Installing pip..."
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
    & "$PythonDir\python.exe" "get-pip.py"
    Remove-Item "get-pip.py"
}

# --- INSTALL PACKAGES ---
Write-Host "? Installing GPU Libraries..."
& "$PythonDir\python.exe" -m pip install -r requirements.txt --no-warn-script-location
Write-Host "? SETUP COMPLETE!" -ForegroundColor Green
Pause
