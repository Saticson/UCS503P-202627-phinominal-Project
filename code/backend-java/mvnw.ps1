[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MavenArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$mavenHome = Join-Path $scriptDir ".mvn\apache-maven-3.9.9"
$mvnCmd = Join-Path $mavenHome "bin\mvn.cmd"

if (-not (Test-Path $mvnCmd)) {
    Write-Host "[INFO] Maven wrapper not found. Downloading Apache Maven 3.9.9..."
    $zipPath = Join-Path $scriptDir ".mvn\maven-bin.zip"
    $mavenUrl = "https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.zip"
    
    Invoke-WebRequest -Uri $mavenUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath (Join-Path $scriptDir ".mvn") -Force
    Remove-Item -Path $zipPath -Force
    Write-Host "[INFO] Maven 3.9.9 initialized successfully."
}

& $mvnCmd $MavenArgs
exit $LASTEXITCODE
