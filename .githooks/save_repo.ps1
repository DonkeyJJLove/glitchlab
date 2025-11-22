param(
  [switch]$Purge,                 # gdy podasz: TRWALE usuwa __pycache__/*.pyc/*.pyo/*.pyd z repo
  [string]$OutDir = ".",          # gdzie zapisać ZIP
  [string]$ZipName = ""           # własna nazwa zipa (opcjonalnie, bez .zip)
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  try {
    $p = (git rev-parse --show-toplevel) 2>$null
    if ($LASTEXITCODE -eq 0 -and $p) { return (Resolve-Path $p).Path }
  } catch {}
  return (Get-Location).Path
}

$repo = Get-RepoRoot
$repoName = Split-Path $repo -Leaf
$ts = (Get-Date).ToString("yyyyMMdd-HHmmss")
if (-not $ZipName -or $ZipName.Trim() -eq "") { $ZipName = "${repoName}-${ts}" }

$OutDir = Resolve-Path $OutDir
$zipPath = Join-Path $OutDir "${ZipName}.zip"

# Wzorce/katalogi do wykluczenia przy pakowaniu
$excludeDirs = @(
  ".git", ".hg", ".svn",
  "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".venv", "env", "venv",
  "build", "dist", ".eggs", "egg-info"
)
$excludeFiles = @("*.pyc","*.pyo","*.pyd",".DS_Store","Thumbs.db")

Write-Host "[pack] repo: $repo"
Write-Host "[pack] out : $zipPath"
if ($Purge) { Write-Host "[pack] mode: PURGE (permanent cleanup in repo)" -ForegroundColor Yellow }
else { Write-Host "[pack] mode: SAFE (no changes in repo)" -ForegroundColor Green }

# 1) Opcjonalny PURGE wewnątrz repo (TRWALE)
if ($Purge) {
  Write-Host "[pack] purging __pycache__ and compiled files in repo..."
  Get-ChildItem -Path $repo -Recurse -Force -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "__pycache__" } |
    ForEach-Object { Remove-Item $_.FullName -Force -Recurse -ErrorAction SilentlyContinue }

  Get-ChildItem -Path $repo -Recurse -Force -File -Include $excludeFiles -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
}

# 2) Zbuduj staging (kopię) bez śmieci i .git
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("repo_pack_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $staging | Out-Null

# użyjemy robocopy do szybkiego mirroru z wykluczeniami
function Join-Args { param([string[]]$arr) $arr -join " " }

$xd = @()
foreach ($d in $excludeDirs) { $xd += "/XD", (Join-Path $repo $d) }
$xf = @()
foreach ($f in $excludeFiles) { $xf += "/XF", $f }

$roboArgs = @(
  "`"$repo`"", "`"$staging`"", "/MIR", "/NFL", "/NDL", "/NP", "/NJH", "/NJS", "/R:1", "/W:1"
) + $xd + $xf

Write-Host "[pack] staging copy..."
$robo = Start-Process -FilePath "robocopy.exe" -ArgumentList $roboArgs -Wait -PassThru -NoNewWindow
# robocopy returns weird codes; treat >=8 as real error
if ($robo.ExitCode -ge 8) { throw "robocopy failed with code $($robo.ExitCode)" }

# 3) Spakuj staging do ZIP
Write-Host "[pack] zipping..."
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
[System.IO.Compression.ZipFile]::CreateFromDirectory($staging, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)

# 4) Sprzątanie stagingu
Remove-Item $staging -Force -Recurse

Write-Host "[pack] done -> $zipPath" -ForegroundColor Cyan
